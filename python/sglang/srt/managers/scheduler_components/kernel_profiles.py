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

A mode is a profile and the scheduler settings that go with it, the unit a
workload-aware controller moves among::

    "modes": {
      "low_latency": {"kernels": "original",
                      "scheduler": {"chunked_prefill_size": 4096,
                                    "prefill_decode_interval": 192}},
      "high_throughput": {"kernels": "low_latency_kernels_or_another_profile",
                          "scheduler": {"chunked_prefill_size": 16384,
                                        "prefill_decode_interval": 0}}}

``switch_mode`` binds the mode's kernels if they differ, then sets exactly its
scheduler settings (the startup value for any it does not name), which take effect
from the next batch. ``live=True`` switches kernels while serving: the engine must
be paused with its running requests retracted (``pause_generation`` with mode
``retract``), so no request holds KV while the graphs are recaptured, and the KV
cache is kept, since an admitted profile computes what the original does.
``Engine.async_switch_mode`` does the pause and the resume.

``"resident_graphs": true`` captures every available profile's decode graphs at
admission and keeps them (``graph_banks``), so a switch rebinds the targets and
selects the profile's graphs between two scheduler steps: no recapture, no pause,
no idle engine. A profile whose graphs cannot be captured, or whose graphs' logits
on the capture inputs differ from the original's beyond ``"bank_check"``, is
unavailable. If capturing one replaces state the graphs captured before read, or
changes what the original graphs compute, the engine drops the banks, recaptures
its original graphs and switches by recapturing.
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
from sglang.srt.model_executor import graph_banks
from sglang.srt.model_executor.process_state_guard import ProcessStateGuard, drift
from sglang.srt.plugins.rpc_registry import register_rpc
from sglang.srt.runtime_context import get_exec, get_parallel

logger = logging.getLogger(__name__)

ORIGINAL = "original"
DEFAULT_MARGIN_GB = 1.0
GRAPH_PHASES = ("decode", "target_verify")
# The original graphs replayed again; a profile's graphs against the original's.
BANK_CHECK = {"stable_rtol": 1e-2, "stable_argmax": 0.02, "rtol": 5e-2, "argmax": 0.1}


def _chunk(value):
    if value is None or int(value) <= 0:
        return None
    return int(value)


def _count(value):
    if int(value) < 0:
        raise ValueError(f"{value} is negative")
    return int(value)


#: Scheduler settings a mode may set, read at every batch, so a new value applies
#: from the next one.
SCHEDULER_SETTINGS = {
    "chunked_prefill_size": _chunk,
    "prefill_decode_interval": _count,
    "prefill_decode_interval_min_bs": _count,
}


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
        self.bank: Optional[dict] = None

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
        self.modes: Dict[str, dict] = {}
        for name, mode in (spec.get("modes") or {}).items():
            kernels = mode.get("kernels", ORIGINAL)
            if kernels != ORIGINAL and kernels not in self.profiles:
                raise ValueError(f"mode {name!r} names no profile {kernels!r}")
            unknown = sorted(set(mode.get("scheduler") or {}) - set(SCHEDULER_SETTINGS))
            if unknown:
                raise ValueError(f"mode {name!r}: no scheduler setting {unknown}")
            settings = {
                key: SCHEDULER_SETTINGS[key](value)
                for key, value in (mode.get("scheduler") or {}).items()
            }
            self.modes[name] = {"kernels": kernels, "scheduler": settings}
        self.startup_settings: Dict[str, object] = {}
        # None after a direct kernel switch, which no mode describes.
        self.active_mode: Optional[str] = None
        self.banks: Optional[graph_banks.GraphBanks] = None
        self.banks_reason: Optional[str] = None

    def mode_available(self, name: str) -> Tuple[bool, Optional[str]]:
        mode = self.modes.get(name)
        if mode is None:
            return False, f"no mode {name!r}"
        kernels = mode["kernels"]
        if kernels != ORIGINAL and not self.profiles[kernels].available:
            return False, f"its kernels {kernels!r} are unavailable"
        return True, None

    def bind(self, name: str) -> None:
        """Bind exactly ``name``'s kernels: its replacements, and every other
        target's original."""
        chosen = {} if name == ORIGINAL else self.profiles[name].replacements
        for spec, target in self.targets.items():
            target.bind(chosen.get(spec, target.original))

    def describe(self) -> dict:
        return {
            "active": self.active,
            "active_mode": self.active_mode,
            "modes": {
                name: {**mode, "available": self.mode_available(name)[0]}
                for name, mode in self.modes.items()
            },
            "broken": self.broken,
            "resident": {
                "on": self.banks is not None,
                "reason": self.banks_reason,
                **(self.banks.describe() if self.banks is not None else {}),
            },
            "profiles": {
                name: {
                    "available": p.available,
                    "reason": p.reason,
                    "targets": p.targets(),
                    "admit_s": p.admit_s,
                    "bank": p.bank,
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
    state.startup_settings = {
        key: getattr(scheduler, key) for key in SCHEDULER_SETTINGS if hasattr(scheduler, key)
    }
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
    if spec.get("resident_graphs"):
        _admit_banks(scheduler, runner, state)
        moved = drift(before, guard.now())
        if moved["flags"] or moved["path"]:
            raise RuntimeError(
                "capturing the kernel profiles' graphs moved process state: "
                + "; ".join(moved["flags"] + moved["path"])
            )
    runner.kernel_profiles = state
    logger.info(
        "kernel profiles: "
        + ", ".join(
            f"{n} ({'resident' if p.bank else 'ready' if p.available else 'unavailable'})"
            for n, p in state.profiles.items()
        )
    )
    return state


def _bank_room(runner, state: KernelProfiles) -> str:
    usage = getattr(runner, "graph_memory_usage", None) or {}
    need = sum(float(usage.get(p, 0.0)) for p in GRAPH_PHASES) + state.margin_gb
    free_gb = torch.cuda.mem_get_info()[0] / 2**30
    if free_gb < need:
        raise RuntimeError(
            f"{free_gb:.2f} GB free; its graphs need {need:.2f} GB "
            f"(the original graphs' and a {state.margin_gb} GB margin)"
        )
    return "ok"


def _admit_banks(scheduler, runner, state: KernelProfiles) -> None:
    """Capture and keep every available profile's decode graphs, on every rank in
    the same order: each capture and replay runs collectives."""
    check = {**BANK_CHECK, **(state.spec.get("bank_check") or {})}

    def support():
        reason = graph_banks.unsupported(runner)
        if reason:
            raise RuntimeError(reason)
        return "ok"

    try:
        _all_ranks(scheduler, support)
    except RuntimeError as exc:
        state.banks_reason = str(exc)[:500]
        logger.warning(f"resident graphs are off: {state.banks_reason}")
        return
    banks = graph_banks.GraphBanks(runner.decode_cuda_graph_runner)
    banks.adopt(ORIGINAL)
    try:
        reference = banks.fingerprints(ORIGINAL)
    except Exception as exc:
        state.banks_reason = f"replaying the original graphs failed: {exc}"[:500]
        logger.warning(f"resident graphs are off: {state.banks_reason}")
        return

    def off(why: str) -> None:
        # Graphs captured earlier may read state that nothing updates any more.
        banks.drop()
        runner.recapture_cuda_graphs()
        torch.cuda.synchronize()
        for profile in state.profiles.values():
            profile.bank = None
        state.banks_reason = why[:500]
        logger.warning(f"resident graphs are off: {state.banks_reason}")

    for name, profile in state.profiles.items():
        if not profile.available:
            continue
        try:
            _all_ranks(scheduler, lambda: _bank_room(runner, state))
            state.bind(name)
            try:
                error, moved = banks.capture(name)
            finally:
                state.bind(ORIGINAL)
            if error is None:
                try:
                    _check_layout(runner, state)
                except RuntimeError as exc:
                    error = str(exc)
            reports = _gather(scheduler, (error, moved))
            moved = sorted({path for _, paths in reports for path in paths})
            if moved:
                return off(f"capturing {name!r} replaced captured state: {moved[:6]}")
            errors = [e for e, _ in reports if e]
            if errors:
                raise RuntimeError(errors[0])
            again = graph_banks.compare(
                reference,
                banks.fingerprints(ORIGINAL),
                rtol=check["stable_rtol"],
                mismatch=check["stable_argmax"],
            )
            changed = [c for c in _gather(scheduler, again) if c]
            if changed:
                return off(f"capturing {name!r} changed the original graphs' logits: {changed[0]}")
            differs = graph_banks.compare(
                reference, banks.fingerprints(name), rtol=check["rtol"], mismatch=check["argmax"]
            )
            differs = [d for d in _gather(scheduler, differs) if d]
            if differs:
                raise RuntimeError(f"its graphs' logits are not the original's: {differs[0]}")
        except Exception as exc:
            if name in banks.banks:
                banks.discard(name)
            profile.available = False
            profile.replacements.clear()
            profile.reason = f"graph bank: {exc}"[:500]
            logger.warning(f"kernel profile {name!r} is unavailable: {profile.reason}")
            continue
        bank = banks.banks[name]
        profile.bank = {
            "graphs": len(bank.graphs),
            "capture_s": round(max(_gather(scheduler, bank.capture_s)), 2),
            "memory_gb": round(max(_gather(scheduler, bank.memory_gb)), 2),
        }
    state.banks = banks


def _paused(scheduler) -> bool:
    pause_state = getattr(scheduler, "pause_state", None)
    return bool(pause_state is not None and pause_state.paused)


def _holds_kv(scheduler) -> bool:
    """Whether a request still holds KV (running, or mid chunked prefill)."""
    return not scheduler.is_fully_idle(ignore_waiting=True)


def _preflight(
    scheduler, runner, state: KernelProfiles, name: str, live: bool = False, resident: bool = False
) -> str:
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
    if resident:
        # Its graphs are captured: the switch is a reference swap between steps.
        return state.active
    if live:
        if not _paused(scheduler):
            raise RuntimeError("a live switch needs generation paused")
        if _holds_kv(scheduler):
            raise RuntimeError(
                "a request still holds KV; pause generation with mode 'retract'"
            )
    elif not scheduler.is_fully_idle():
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


def _rebind(scheduler, runner, state: KernelProfiles, name: str, live: bool = False) -> float:
    def drain():
        # A live switch keeps the retracted requests' cached prefixes: an admitted
        # profile computes what the original does.
        if not live and not scheduler.flush_cache():
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


def _switch_resident(scheduler, runner, state: KernelProfiles, name: str, live: bool) -> dict:
    started = time.perf_counter()

    def preflight():
        previous = _preflight(scheduler, runner, state, name, resident=True)
        if previous == ORIGINAL and name != ORIGINAL:
            state.guard.reference()
        return previous

    previous = _all_ranks(scheduler, preflight)

    def select(target):
        def flip():
            state.bind(target)
            state.banks.activate(target)
            return target

        return flip

    if name != previous:
        try:
            _all_ranks(scheduler, select(name))
        except Exception as rejected:
            try:
                _all_ranks(scheduler, select(previous))
            except Exception as rollback:
                state.broken = True
                raise SystemExit(
                    f"kernel profile switch and rollback both failed: {rollback}"
                ) from rejected
            raise RuntimeError(f"kept kernel profile {previous!r}: {rejected}") from rejected
        state.active = name
    if name == ORIGINAL and previous != ORIGINAL:
        _verify_original(scheduler, state, "original kernels restored")
    return {
        "active": name,
        "previous": previous,
        "live": live,
        "resident": True,
        "capture_s": None,
        "total_s": time.perf_counter() - started,
    }


def _switch_kernels(scheduler, runner, state: KernelProfiles, name: str, live: bool) -> dict:
    if state.banks is not None:
        return _switch_resident(scheduler, runner, state, name, live)
    started = time.perf_counter()

    def preflight():
        previous = _preflight(scheduler, runner, state, name, live)
        if previous == ORIGINAL and name != ORIGINAL:
            # The engine has served on its original kernels and is about to leave
            # them: this is the state coming back must find again.
            state.guard.reference()
        return previous

    previous = _all_ranks(scheduler, preflight)
    free_before = min(_gather(scheduler, torch.cuda.mem_get_info()[0] / 2**30))
    capture_s = None
    if name != previous:
        capture_s = _rebind(scheduler, runner, state, name, live)
    if name == ORIGINAL and previous != ORIGINAL:
        _verify_original(scheduler, state, "original kernels restored")
    return {
        "active": name,
        "previous": previous,
        "live": live,
        "capture_s": capture_s,
        "total_s": time.perf_counter() - started,
        "free_gb_before": round(free_before, 2),
        "free_gb_after": round(
            min(_gather(scheduler, torch.cuda.mem_get_info()[0] / 2**30)), 2
        ),
    }


@register_rpc("switch_kernel_profile")
def switch_kernel_profile(scheduler, *, name: str, live: bool = False) -> str:
    state = _state(scheduler.tp_worker.model_runner)
    result = _switch_kernels(scheduler, scheduler.tp_worker.model_runner, state, name, live)
    if result["active"] != result["previous"]:
        state.active_mode = None
    state.switches.append({**result, "at": time.time()})
    return json.dumps(result)


@register_rpc("plan_mode")
def plan_mode(scheduler, *, name: str) -> str:
    """What switching to mode ``name`` takes now, without changing anything."""
    state = _state(scheduler.tp_worker.model_runner)
    available, reason = state.mode_available(name)
    if not available:
        raise ValueError(f"mode {name!r} is unavailable: {reason}")
    return json.dumps(
        {
            "mode": name,
            "kernels_change": state.modes[name]["kernels"] != state.active,
            # Resident graphs: a kernel change needs no pause either.
            "resident": state.banks is not None,
            "paused": _paused(scheduler),
            "holds_kv": _holds_kv(scheduler),
        }
    )


@register_rpc("switch_mode")
def switch_mode(scheduler, *, name: str, live: bool = False) -> str:
    started = time.perf_counter()
    runner = scheduler.tp_worker.model_runner
    state = _state(runner)
    available, reason = state.mode_available(name)
    if not available:
        raise ValueError(f"mode {name!r} is unavailable: {reason}")
    mode = state.modes[name]
    kernels = None
    if mode["kernels"] != state.active:
        kernels = _switch_kernels(scheduler, runner, state, mode["kernels"], live)
    settings = {**state.startup_settings, **mode["scheduler"]}

    def apply():
        for key, value in settings.items():
            setattr(scheduler, key, value)
        if "prefill_decode_interval" in settings:
            # A shorter interval must not wait out the remainder of a longer one.
            scheduler._prefill_decode_interval_remaining = min(
                getattr(scheduler, "_prefill_decode_interval_remaining", 0),
                settings["prefill_decode_interval"],
            )
        return sorted(settings.items())

    _all_ranks(scheduler, apply)
    previous_mode = state.active_mode
    state.active_mode = name
    result = {
        "mode": name,
        "previous_mode": previous_mode,
        "kernels": kernels,
        "scheduler": settings,
        "total_s": time.perf_counter() - started,
    }
    state.switches.append({**result, "at": time.time()})
    return json.dumps(result)


@register_rpc("list_kernel_profiles")
def list_kernel_profiles(scheduler) -> str:
    return json.dumps(_state(scheduler.tp_worker.model_runner).describe())
