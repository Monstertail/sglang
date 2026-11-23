"""The scheduler RPCs: swap the model's forward and recapture its decode CUDA graphs.

``hot_reload_forward`` makes every scheduler load ``path``, call its
``make_forward(original_forward)``, serve the callable it returns and recapture
the decode CUDA graphs. ``restore_forward`` serves the original forward again.
Prefill must run eager (``--disable-prefill-cuda-graph``): prefill CUDA graphs
replay the decoder layers and never call ``model.forward``. Wrap the new forward
with ``functools.wraps(original_forward)``; callers inspect its signature.

Every TP rank checks the same preconditions and agrees before any rank changes
state, and a rank that cannot get back to a captured graph stops the engine
instead of serving.

The loaded code runs inside the engine's processes, so a swap refuses what
restoring the original callable cannot undo:

- a ``make_forward`` that moves, reshapes or replaces a parameter;
- process state (see ``ProcessStateGuard``) that differs once the original
  forward is back, whether it was restored or kept after a refused candidate.
"""

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

from .process_state_guard import ProcessStateGuard

ORIGINAL = "original"


def _enabled() -> bool:
    return os.environ.get("SGLANG_FORWARD_HOT_RELOAD") == "1"


def _parallel():
    from sglang.srt.runtime_context import get_parallel

    return get_parallel()


def parameter_layout(model) -> tuple:
    return tuple(
        (name, p.data_ptr(), tuple(p.shape), tuple(p.stride()), p.dtype)
        for name, p in model.named_parameters()
    )


def _module_name(sha256: str) -> str:
    return f"_sglang_forward_{sha256}"


def load_make_forward(path: str, sha256: str):
    source = Path(path).read_bytes()
    if hashlib.sha256(source).hexdigest() != sha256:
        raise ValueError(f"{path} does not match sha256 {sha256}")
    name = _module_name(sha256)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(source, path, "exec"), module.__dict__)
    return module.make_forward


class ForwardSlot:
    """The forward state of one model runner."""

    def __init__(self, model):
        self.original = model.forward
        self.active = ORIGINAL
        self.layout = parameter_layout(model)
        self.guard = ProcessStateGuard(torch)
        # True while no verified captured graph matches the served forward.
        self.broken = False


def _slot(runner) -> ForwardSlot:
    slot = getattr(runner, "forward_hot_reload_slot", None)
    if slot is None:
        slot = runner.forward_hot_reload_slot = ForwardSlot(runner.model)
    return slot


def _has_prefill_graph(runner) -> bool:
    graph = getattr(runner, "prefill_cuda_graph_runner", None)
    return graph not in (None, getattr(runner, "eager_runner", None))


def _recapture(runner) -> None:
    if _has_prefill_graph(runner):
        # Full and breakable prefill graphs replay the decoder layers, not
        # model.forward, and piecewise ones are compiled from it once.
        raise NotImplementedError(
            "cannot recapture prefill CUDA graphs; launch with "
            "--disable-prefill-cuda-graph"
        )
    runner.recapture_decode_cuda_graph()


def _all_ranks(scheduler, check):
    """Run ``check`` on every TP rank; raise on every rank if one raised or the
    ranks returned different values, so that no rank goes on alone."""
    try:
        value, error = check(), None
    except Exception as exc:
        value, error = None, f"{type(exc).__name__}: {exc}"
    group = scheduler.tp_group.cpu_group
    reports = [None] * dist.get_world_size(group)
    dist.all_gather_object(reports, (value, error), group=group)
    if any(error for _, error in reports):
        raise RuntimeError(f"rank checks failed: {reports}")
    if any(report != reports[0] for report in reports):
        raise RuntimeError(f"ranks disagree: {reports}")
    return value


def _report(scheduler, status_dir, result: dict) -> str:
    text = json.dumps(result)
    if status_dir is not None:
        rank = dist.get_rank(scheduler.tp_group.cpu_group)
        dest = Path(status_dir) / f"rank-{rank}.json"
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(text)
        tmp.replace(dest)
    return text


def _preflight(scheduler, runner, slot):
    if not _enabled():
        raise RuntimeError(
            "set SGLANG_FORWARD_HOT_RELOAD=1 in the engine's environment"
        )
    parallel = _parallel()
    if parallel.attn_dp_size > 1 or parallel.pp_size > 1:
        raise RuntimeError("forward hot reload supports tensor parallelism only")
    if _has_prefill_graph(runner):
        # Prefill graphs replay the decoder layers and never call model.forward,
        # so a swapped forward would be skipped on prefill.
        raise RuntimeError(
            "forward hot reload needs eager prefill: launch with "
            "--disable-prefill-cuda-graph"
        )
    if slot.broken:
        raise RuntimeError("an earlier swap left this engine without a known forward")
    if not scheduler.is_fully_idle():
        raise RuntimeError("the engine is serving requests")
    return slot.active


def _check_layout(runner, slot):
    if parameter_layout(runner.model) != slot.layout:
        raise RuntimeError("parameter storage changed")


def _verify_original(scheduler, slot, what: str) -> None:
    # A candidate's module code has run even when the candidate was refused, so
    # the original forward must not go on serving in a state it left behind.
    try:
        _all_ranks(scheduler, slot.guard.verify)
    except RuntimeError as exc:
        slot.broken = True
        raise SystemExit(f"{what} but {exc}") from exc


def _swap(scheduler, runner, slot, forward, name) -> float:
    def drain():
        if not scheduler.flush_cache():
            raise RuntimeError("could not flush the KV cache")
        torch.cuda.synchronize()

    _all_ranks(scheduler, drain)

    previous_name, previous = slot.active, runner.model.forward
    capture_s = None

    def recapture():
        nonlocal capture_s
        started = time.perf_counter()
        _recapture(runner)
        torch.cuda.synchronize()
        capture_s = time.perf_counter() - started
        _check_layout(runner, slot)

    slot.broken = True
    runner.model.forward = forward
    try:
        _all_ranks(scheduler, recapture)
    except Exception as rejected:
        runner.model.forward = previous
        try:
            _all_ranks(scheduler, recapture)
        except Exception as rollback:
            # SystemExit escapes handle_rpc_request's handler: a rank without a
            # captured graph for its forward must not return to generation.
            raise SystemExit(
                f"forward swap and rollback both failed: {rollback}"
            ) from rejected
        slot.broken = False
        if previous_name == ORIGINAL:
            _verify_original(
                scheduler, slot, "kept the original forward after a rejected capture"
            )
        raise RuntimeError(f"kept the previous forward: {rejected}") from rejected
    slot.broken = False
    slot.active = name
    if previous_name not in (ORIGINAL, name):
        sys.modules.pop(_module_name(previous_name), None)
    return capture_s


def hot_reload_forward(scheduler, *, path: str, sha256: str, status_dir=None) -> str:
    runner = scheduler.tp_worker.model_runner
    slot = _slot(runner)

    def preflight():
        active = _preflight(scheduler, runner, slot)
        if active == ORIGINAL:
            # The engine has served on its original state and is about to leave
            # it: this is the state a later restore must find again.
            slot.guard.reference()
        return active

    _all_ranks(scheduler, preflight)
    forward = None

    def build():
        nonlocal forward
        forward = load_make_forward(path, sha256)(slot.original)
        if not callable(forward):
            raise TypeError("make_forward must return a callable")

    try:
        _all_ranks(scheduler, build)
    except Exception:
        if sha256 != slot.active:
            sys.modules.pop(_module_name(sha256), None)
        if slot.active == ORIGINAL:
            _verify_original(
                scheduler, slot, "kept the original forward after a refused candidate"
            )
        raise
    finally:
        try:
            _all_ranks(scheduler, lambda: _check_layout(runner, slot))
        except RuntimeError as exc:
            slot.broken = True
            raise SystemExit(f"make_forward changed the weights: {exc}") from exc
    capture_s = _swap(scheduler, runner, slot, forward, sha256)
    return _report(scheduler, status_dir, {"active": sha256, "capture_s": capture_s})


def restore_forward(scheduler, *, status_dir=None) -> str:
    runner = scheduler.tp_worker.model_runner
    slot = _slot(runner)
    _all_ranks(scheduler, lambda: _preflight(scheduler, runner, slot))
    capture_s = None
    if slot.active != ORIGINAL:
        capture_s = _swap(scheduler, runner, slot, slot.original, ORIGINAL)
    _verify_original(scheduler, slot, "original forward restored")
    moved = slot.guard.moved()
    return _report(
        scheduler,
        status_dir,
        {
            "active": ORIGINAL,
            "capture_s": capture_s,
            "env_moved": [] if moved is None else moved["env"],
        },
    )
