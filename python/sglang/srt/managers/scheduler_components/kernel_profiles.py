"""Move a running engine among declared sets of kernels (``--kernel-profiles``).

A kernel profile is a named set of bindings. Each one replaces a kernel entry point
that its callers look up at call time, such as
``aiter.ops.flydsl.moe_kernels:flydsl_moe_stage1``, with another implementation
that is already importable: binaries built for one kind of traffic (low latency,
high throughput) that the server moves among without a restart.
``Engine.switch_kernel_profile(name)`` rebinds them and recaptures the CUDA graphs
that replay ``model.forward``. ``original`` is every target's own entry point.

The file ``--kernel-profiles`` names is JSON::

    {"path": ["/opt/kernels"],            # prepended to sys.path (optional)
     "memory_margin_gb": 1.0,             # free memory a switch keeps (optional)
     "profiles": {
       "low_latency": {
         "bindings": [
           {"target": "pkg.mod:attr", "impl": "kpkg.mod:attr"},
           {"target": "pkg.mod:other", "factory": "kpkg.mod:make"}],
         "warmup": "kpkg.mod:warmup",     # warmup(pairs): compile its binaries now
         "check": "kpkg.mod:check"}}}     # check(pairs): raise if one is wrong

``make(original)`` returns the replacement; ``pairs`` maps each target to
``(original, replacement)``.

Admission runs once per scheduler at startup: every profile is imported, resolved,
warmed up and checked on every TP rank. A profile that fails on any rank is
unavailable, with the reason, and the server serves ``original``; so a switch
compiles nothing and runs no unchecked binary. Admitting a profile must not move
process state (``ProcessStateGuard``) or rebind a target, or the server does not
start.

A switch is refused unless every rank agrees that the engine is idle, that its free
device memory covers the last capture of its graphs plus the margin, and that no
earlier switch left it broken. A capture that fails puts the previous bindings back
on every rank and recaptures; if that fails too the engine stops. Coming back to
``original`` checks the process state, and the weights' storage must not change.
Prefill must run eager (``--disable-prefill-cuda-graph``): prefill graphs are not
recaptured and would keep the kernels they were captured with. A speculative
draft's graphs are not recaptured either.
"""

import importlib
import json
import logging
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from sglang.srt.managers.scheduler_components.forward_hot_reload import (
    _all_ranks,
    parameter_layout,
)
from sglang.srt.model_executor.process_state_guard import ProcessStateGuard, drift
from sglang.srt.plugins.rpc_registry import register_rpc
from sglang.srt.runtime_context import get_exec, get_parallel

logger = logging.getLogger(__name__)

ORIGINAL = "original"
DEFAULT_MARGIN_GB = 1.0
GRAPH_PHASES = ("decode", "target_verify")


def _resolve(spec: str):
    """``"pkg.mod:attr.sub"`` -> (owner of the last attribute, its name, its value)."""
    module_name, sep, attrs = spec.partition(":")
    if not sep or not module_name or not attrs:
        raise ValueError(f"{spec!r} is not module:attribute")
    owner = importlib.import_module(module_name)
    *parents, name = attrs.split(".")
    for part in parents:
        owner = getattr(owner, part)
    if not hasattr(owner, name):
        raise AttributeError(f"{spec!r}: {module_name} has no {attrs}")
    return owner, name, getattr(owner, name)


class Target:
    """One rebound entry point and the callable it had at admission."""

    def __init__(self, spec: str):
        self.spec = spec
        self.owner, self.name, self.original = _resolve(spec)
        if not callable(self.original):
            raise TypeError(f"{spec} is not callable")

    def current(self):
        return getattr(self.owner, self.name)

    def bind(self, value) -> None:
        setattr(self.owner, self.name, value)


class Profile:
    def __init__(self, name: str, spec: dict):
        self.name = name
        self.spec = spec
        self.replacements: Dict[str, object] = {}
        self.available = False
        self.reason: Optional[str] = None
        self.admit_s: Optional[float] = None

    def targets(self) -> List[str]:
        return [b["target"] for b in self.spec.get("bindings", [])]

    def build(self, targets: Dict[str, Target]) -> None:
        bindings = self.spec.get("bindings") or []
        if not bindings:
            raise ValueError("a profile binds at least one target")
        seen = set()
        for b in bindings:
            target = targets[b["target"]]
            if target.spec in seen:
                raise ValueError(f"{target.spec} is bound twice")
            seen.add(target.spec)
            if ("impl" in b) == ("factory" in b):
                raise ValueError(f"{target.spec}: give exactly one of impl or factory")
            if "impl" in b:
                value = _resolve(b["impl"])[2]
            else:
                value = _resolve(b["factory"])[2](target.original)
            if not callable(value):
                raise TypeError(f"{target.spec}: the replacement is not callable")
            self.replacements[target.spec] = value

    def pairs(self, targets: Dict[str, Target]) -> Dict[str, Tuple[object, object]]:
        return {
            spec: (targets[spec].original, value)
            for spec, value in self.replacements.items()
        }

    def run_hook(self, key: str, targets: Dict[str, Target]) -> None:
        if self.spec.get(key):
            _resolve(self.spec[key])[2](self.pairs(targets))


class KernelProfiles:
    """The profiles one model runner serves, and which one it serves."""

    def __init__(self, spec: dict, layout: tuple):
        self.spec = spec
        self.margin_gb = float(spec.get("memory_margin_gb", DEFAULT_MARGIN_GB))
        self.profiles = {
            name: Profile(name, p) for name, p in (spec.get("profiles") or {}).items()
        }
        if ORIGINAL in self.profiles:
            raise ValueError(f"{ORIGINAL!r} names the targets' own entry points")
        self.targets: Dict[str, Target] = {}
        self.active = ORIGINAL
        self.layout = layout
        self.guard = ProcessStateGuard(torch)
        # True while no verified captured graph matches the bound kernels.
        self.broken = False
        self.switches: List[dict] = []

    def bind(self, name: str) -> None:
        """Bind exactly ``name``'s kernels: its replacements, and every other
        target's original."""
        chosen = {} if name == ORIGINAL else self.profiles[name].replacements
        for spec, target in self.targets.items():
            target.bind(chosen.get(spec, target.original))

    def describe(self) -> dict:
        return {
            "active": self.active,
            "broken": self.broken,
            "profiles": {
                name: {
                    "available": p.available,
                    "reason": p.reason,
                    "targets": p.targets(),
                    "admit_s": p.admit_s,
                }
                for name, p in self.profiles.items()
            },
            "switches": self.switches[-20:],
        }


def _state(runner) -> KernelProfiles:
    state = getattr(runner, "kernel_profiles", None)
    if state is None:
        raise RuntimeError("launch the server with --kernel-profiles")
    return state


def _gather(scheduler, value) -> list:
    group = scheduler.tp_group.cpu_group
    values = [None] * dist.get_world_size(group)
    dist.all_gather_object(values, value, group=group)
    return values


def admit(scheduler, path: Optional[str] = None) -> KernelProfiles:
    """Load ``--kernel-profiles`` and admit each profile on every TP rank."""
    path = path or get_exec().features.kernel_profiles
    with open(path) as f:
        spec = json.load(f)
    for entry in reversed(spec.get("path") or []):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    runner = scheduler.tp_worker.model_runner
    state = KernelProfiles(spec, parameter_layout(runner.model))
    guard = ProcessStateGuard(torch)
    before = guard.now()

    def resolve_targets():
        for profile in state.profiles.values():
            for spec_ in profile.targets():
                if spec_ not in state.targets:
                    state.targets[spec_] = Target(spec_)
        return sorted(state.targets)

    _all_ranks(scheduler, resolve_targets)
    for name, profile in state.profiles.items():

        def build():
            started = time.perf_counter()
            profile.build(state.targets)
            rebound = [t.spec for t in state.targets.values() if t.current() is not t.original]
            if rebound:
                raise RuntimeError(f"importing it rebound {rebound}")
            profile.run_hook("warmup", state.targets)
            profile.run_hook("check", state.targets)
            profile.admit_s = time.perf_counter() - started
            return "ok"

        try:
            _all_ranks(scheduler, build)
            profile.available = True
        except Exception as exc:  # an unavailable profile, not a server that fails to start
            profile.replacements.clear()
            profile.reason = str(exc)[:500]
            logger.warning(f"kernel profile {name!r} is unavailable: {profile.reason}")
        state.bind(ORIGINAL)
        moved = drift(before, guard.now())
        if moved["flags"] or moved["path"]:
            raise RuntimeError(
                f"admitting kernel profile {name!r} moved process state: "
                + "; ".join(moved["flags"] + moved["path"])
            )
    runner.kernel_profiles = state
    logger.info(
        "kernel profiles: "
        + ", ".join(
            f"{n} ({'ready' if p.available else 'unavailable'})"
            for n, p in state.profiles.items()
        )
    )
    return state


def _preflight(scheduler, runner, state: KernelProfiles, name: str) -> str:
    parallel = get_parallel()
    if parallel.attn_dp_size > 1 or parallel.pp_size > 1:
        raise RuntimeError("kernel profiles support tensor parallelism only")
    if runner.has_prefill_cuda_graph():
        raise RuntimeError(
            "kernel profiles need eager prefill: launch with "
            "--disable-prefill-cuda-graph"
        )
    slot = getattr(runner, "forward_hot_reload_slot", None)
    if state.broken or (slot is not None and slot.broken):
        raise RuntimeError("an earlier swap left this engine without a known forward")
    if name != ORIGINAL:
        profile = state.profiles.get(name)
        if profile is None:
            raise ValueError(f"no kernel profile {name!r}")
        if not profile.available:
            raise ValueError(f"kernel profile {name!r} is unavailable: {profile.reason}")
    if not scheduler.is_fully_idle():
        raise RuntimeError("the engine is serving requests; drain it first")
    usage = getattr(runner, "graph_memory_usage", None) or {}
    need = sum(float(usage.get(p, 0.0)) for p in GRAPH_PHASES) + state.margin_gb
    free_gb = torch.cuda.mem_get_info()[0] / 2**30
    if free_gb < need:
        raise RuntimeError(
            f"{free_gb:.2f} GB free; recapturing needs {need:.2f} GB "
            f"(the last capture's graphs and a {state.margin_gb} GB margin)"
        )
    return state.active


def _check_layout(runner, state: KernelProfiles) -> None:
    if parameter_layout(runner.model) != state.layout:
        raise RuntimeError("parameter storage changed")


def _verify_original(scheduler, state: KernelProfiles, what: str) -> None:
    # The profiles' code has run in this process, so the original kernels must not
    # serve in a state it left behind.
    try:
        _all_ranks(scheduler, state.guard.verify)
    except RuntimeError as exc:
        state.broken = True
        raise SystemExit(f"{what} but {exc}") from exc


def _rebind(scheduler, runner, state: KernelProfiles, name: str) -> float:
    def drain():
        if not scheduler.flush_cache():
            raise RuntimeError("could not flush the KV cache")
        torch.cuda.synchronize()

    _all_ranks(scheduler, drain)
    previous = state.active
    capture_s = None

    def recapture():
        nonlocal capture_s
        started = time.perf_counter()
        runner.recapture_cuda_graphs()
        torch.cuda.synchronize()
        capture_s = time.perf_counter() - started
        _check_layout(runner, state)

    state.broken = True
    state.bind(name)
    try:
        _all_ranks(scheduler, recapture)
    except Exception as rejected:
        state.bind(previous)
        try:
            _all_ranks(scheduler, recapture)
        except Exception as rollback:
            # SystemExit escapes handle_rpc_request's handler: a rank without a
            # captured graph for its kernels must not return to generation.
            raise SystemExit(
                f"kernel profile switch and rollback both failed: {rollback}"
            ) from rejected
        state.broken = False
        if previous == ORIGINAL:
            _verify_original(
                scheduler, state, "kept the original kernels after a rejected capture"
            )
        raise RuntimeError(f"kept kernel profile {previous!r}: {rejected}") from rejected
    state.broken = False
    state.active = name
    return capture_s


@register_rpc("switch_kernel_profile")
def switch_kernel_profile(scheduler, *, name: str) -> str:
    started = time.perf_counter()
    runner = scheduler.tp_worker.model_runner
    state = _state(runner)

    def preflight():
        previous = _preflight(scheduler, runner, state, name)
        if previous == ORIGINAL and name != ORIGINAL:
            # The engine has served on its original kernels and is about to leave
            # them: this is the state coming back must find again.
            state.guard.reference()
        return previous

    previous = _all_ranks(scheduler, preflight)
    free_before = min(_gather(scheduler, torch.cuda.mem_get_info()[0] / 2**30))
    capture_s = None
    if name != previous:
        capture_s = _rebind(scheduler, runner, state, name)
    if name == ORIGINAL and previous != ORIGINAL:
        _verify_original(scheduler, state, "original kernels restored")
    result = {
        "active": name,
        "previous": previous,
        "capture_s": capture_s,
        "total_s": time.perf_counter() - started,
        "free_gb_before": round(free_before, 2),
        "free_gb_after": round(
            min(_gather(scheduler, torch.cuda.mem_get_info()[0] / 2**30)), 2
        ),
    }
    state.switches.append({**result, "at": time.time()})
    return json.dumps(result)


@register_rpc("list_kernel_profiles")
def list_kernel_profiles(scheduler) -> str:
    return json.dumps(_state(scheduler.tp_worker.model_runner).describe())
