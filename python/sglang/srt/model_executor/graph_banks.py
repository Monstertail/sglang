"""Decode graphs kept per kernel set, so that switching kernels selects captured
graphs instead of capturing them (``"resident_graphs"`` in ``--kernel-profiles``).

A bank is what one capture of a decode graph runner writes and its graphs read back
at replay: the graph backend's graphs and outputs, the attention backends' captured
metadata (``cuda_graph_metadata_of_bucket_and_bs``), the metadata glue graphs and
the ragged verify layouts. The banks share the runner's static input buffers, its
logits buffer and the graph memory pool, as the graphs of different batch sizes do:
one graph is replayed at a time, on one stream.

Capturing a bank must leave the others alone. Each bank holds every tensor reachable
from the runner, its graph backend and its attention backends when it is made, so
no memory its graphs read is freed, and ``capture`` reports by path each of those
tensors it replaced (drift): a graph captured earlier still reads the old one, which
nothing updates any more. A caller that sees drift drops the banks and recaptures.

Activating a bank swaps references. It happens between two scheduler steps; a step
prepares its attention metadata in the active bank's objects (``load_batch``) and
replays that bank's graph.
"""

import contextlib
import functools
import logging
import time
import types
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode

logger = logging.getLogger(__name__)

METADATA = "cuda_graph_metadata_of_bucket_and_bs"
# Attributes a bank owns, state set per step, and back references, by root.
_RUNNER_SKIP = frozenset(
    {
        "model_runner",
        "backend",
        "attn_backend",
        "_metadata_glue",
        "_early_verify_metadata",
        "_captured_ragged_layouts",
        "_staged_inputs",
        "_prefilled_inputs",
        "_profiler",
    }
)
_BACKEND_SKIP = frozenset({"_graphs", "_outputs", "_capture_inputs", "_cuda_graph_runner"})
# candidate_masks: handed from one layer to the later ones within a forward; a graph
# makes and reads its own.
_ATTN_SKIP = frozenset(
    {
        METADATA,
        "forward_metadata",
        "_current_capture_raw",
        "candidate_masks",
        "model_runner",
        "attn_backend_list",
    }
)
_GLUE_SKIP = frozenset({"_states"})
_EARLY_SKIP = frozenset({"_staged"})
_MAX_DEPTH = 4
_MAX_ITEMS = 4096
_OPAQUE = (
    str,
    bytes,
    int,
    float,
    bool,
    type(None),
    type,
    types.ModuleType,
    types.FunctionType,
    types.MethodType,
    types.BuiltinFunctionType,
    functools.partial,
    torch.nn.Module,
    torch.dtype,
    torch.device,
    torch.cuda.Stream,
    torch.cuda.Event,
    torch.cuda.CUDAGraph,
)


class Bank:
    def __init__(self, *, graphs, outputs, capture_inputs, metadata, glue, ragged, keep):
        self.graphs: dict = graphs
        self.outputs: dict = outputs
        self.capture_inputs: Optional[dict] = capture_inputs
        self.metadata: List[dict] = metadata
        self.glue: Optional[dict] = glue
        self.ragged: Optional[dict] = ragged
        # Every tensor the bank's graphs may read outside its own dicts.
        self.keep: List[torch.Tensor] = keep
        self.capture_s: Optional[float] = None
        self.memory_gb: Optional[float] = None


def _leaves(attn_backend) -> list:
    leaves, queue = [], [attn_backend]
    while queue:
        backend = queue.pop(0)
        if backend is None or any(backend is leaf for leaf in leaves):
            continue
        leaves.append(backend)
        queue.extend(getattr(backend, "attn_backend_list", None) or [])
    return leaves


def _fields(value):
    fields = getattr(value, "__dict__", None)
    if fields is not None:
        return list(fields.items())
    names = getattr(value, "__struct_fields__", None) or getattr(value, "__slots__", ())
    if isinstance(names, str):
        names = (names,)
    return [(name, getattr(value, name)) for name in names if hasattr(value, name)]


def _tensors(roots) -> Dict[str, torch.Tensor]:
    """Every tensor reachable from ``roots`` (name, object, attributes to skip), by path."""
    found: Dict[str, torch.Tensor] = {}
    seen = set()

    def visit(value, path, depth):
        if isinstance(value, torch.Tensor):
            found[path] = value
            return
        if depth >= _MAX_DEPTH or isinstance(value, _OPAQUE) or id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, dict):
            for key, item in list(value.items())[:_MAX_ITEMS]:
                visit(item, f"{path}[{key!r}]", depth + 1)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value[:_MAX_ITEMS]):
                visit(item, f"{path}[{index}]", depth + 1)
        elif not isinstance(value, (set, frozenset)):
            for name, item in _fields(value):
                visit(item, f"{path}.{name}", depth + 1)

    for name, root, skip in roots:
        seen.add(id(root))
        for attr, value in _fields(root):
            if attr not in skip:
                visit(value, f"{name}.{attr}", 1)
    return found


def _moved(before: Dict[str, torch.Tensor], after: Dict[str, torch.Tensor]) -> List[str]:
    return sorted(
        path
        for path, tensor in before.items()
        if path not in after or after[path].data_ptr() != tensor.data_ptr()
    )


def _reset(bank: Bank) -> None:
    for graph in bank.graphs.values():
        reset = getattr(graph, "reset", None)
        if reset is not None:
            reset()
    bank.graphs.clear()
    bank.outputs.clear()
    bank.keep = []


def _end_capture_variant() -> None:
    # A capture that raised leaves its attention variant set, which eager kernels read.
    try:
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            _set_capture_attention_variant,
        )
    except ImportError:
        return
    _set_capture_attention_variant(None)


def _logits(output) -> Optional[torch.Tensor]:
    logits = getattr(output, "next_token_logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(output, torch.Tensor):
        return output
    values = output if isinstance(output, (list, tuple)) else [v for _, v in _fields(output)]
    for value in values:
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            return value
    return None


def _fingerprint(output) -> Optional[tuple]:
    logits = _logits(output)
    if logits is None:
        return None
    rows = logits.detach().float().reshape(max(1, logits.shape[0] if logits.ndim else 1), -1)
    return (
        bool(torch.isfinite(rows).all()),
        rows.norm(dim=-1).double().cpu(),
        rows.argmax(dim=-1).cpu(),
    )


def compare(reference: Dict[str, tuple], other: Dict[str, tuple], *, rtol: float, mismatch: float) -> Optional[str]:
    """Why ``other``'s logits are not ``reference``'s, or None."""
    if sorted(reference) != sorted(other):
        return "the banks replay different graph keys"
    for key, ref in reference.items():
        got = other[key]
        if (ref is None) != (got is None):
            return f"{key}: only one bank returns logits"
        if ref is None:
            continue
        if not (ref[0] and got[0]):
            return f"{key}: non-finite logits"
        rel = float(((got[1] - ref[1]).abs() / ref[1].abs().clamp_min(1e-6)).max())
        flips = float((got[2] != ref[2]).double().mean())
        if rel > rtol or flips > mismatch:
            return f"{key}: logit norms differ by {rel:.3g}, argmax in {flips:.1%} of rows"
    return None


def unsupported(model_runner) -> Optional[str]:
    """Why this model runner's decode graphs cannot be banked, or None."""
    runner = getattr(model_runner, "decode_cuda_graph_runner", None)
    if runner is None or runner is getattr(model_runner, "eager_runner", None):
        return "the engine captures no decode graphs"
    backend = getattr(runner, "backend", None)
    if not (
        isinstance(getattr(backend, "_graphs", None), dict)
        and isinstance(getattr(backend, "_outputs", None), dict)
        and callable(getattr(backend, "replay", None))
    ):
        return f"the {type(backend).__name__} graph backend keeps no graphs per shape"
    if not callable(getattr(runner, "capture", None)):
        return f"{type(runner).__name__} cannot capture again"
    if getattr(runner, "enable_two_batch_overlap", False):
        return "two-batch overlap keeps its own capture state"
    if getattr(model_runner, "lora_manager", None) is not None:
        return "LoRA keeps its own capture state"
    spec = getattr(model_runner, "spec_algorithm", None)
    if spec is not None and spec.is_speculative() and not getattr(spec, "is_dspark", lambda: False)():
        return f"{spec} may plan a verify's metadata before its step; banks are made for decode and DSpark verify"
    try:
        from sglang.srt.model_executor.runner_backend_utils.replay_hooks import (
            has_replay_hooks,
        )

        if has_replay_hooks():
            return "replay hooks replay a step against its batch"
    except ImportError:
        pass
    for leaf in _leaves(getattr(runner, "attn_backend", None)):
        if not isinstance(getattr(leaf, METADATA, None), dict):
            return f"attention backend {type(leaf).__name__} keeps capture metadata the banks do not separate"
    return None


class GraphBanks:
    """The banks of one decode graph runner, and which one it replays."""

    def __init__(self, graph_runner):
        self.runner = graph_runner
        self.leaves = _leaves(graph_runner.attn_backend)
        self.banks: Dict[str, Bank] = {}
        self.active: Optional[str] = None

    def _roots(self):
        runner = self.runner
        roots = [("runner", runner, _RUNNER_SKIP), ("backend", runner.backend, _BACKEND_SKIP)]
        roots += [(f"attn[{i}]", leaf, _ATTN_SKIP) for i, leaf in enumerate(self.leaves)]
        glue = getattr(runner, "_metadata_glue", None)
        if glue is not None:
            roots.append(("glue", glue, _GLUE_SKIP))
        early = getattr(runner, "_early_verify_metadata", None)
        if early is not None:
            roots.append(("early", early, _EARLY_SKIP))
        return roots

    def _installed(self, keep) -> Bank:
        runner, backend = self.runner, self.runner.backend
        glue = getattr(runner, "_metadata_glue", None)
        return Bank(
            graphs=backend._graphs,
            outputs=backend._outputs,
            capture_inputs=getattr(backend, "_capture_inputs", None),
            metadata=[getattr(leaf, METADATA) for leaf in self.leaves],
            glue=None if glue is None else glue._states,
            ragged=getattr(runner, "_captured_ragged_layouts", None),
            keep=keep,
        )

    def _install(self, bank: Bank) -> None:
        runner, backend = self.runner, self.runner.backend
        backend._graphs, backend._outputs = bank.graphs, bank.outputs
        if bank.capture_inputs is not None:
            backend._capture_inputs = bank.capture_inputs
        for leaf, metadata in zip(self.leaves, bank.metadata):
            setattr(leaf, METADATA, metadata)
        glue = getattr(runner, "_metadata_glue", None)
        if glue is not None:
            glue._states = bank.glue
        if bank.ragged is not None:
            runner._captured_ragged_layouts = bank.ragged

    def adopt(self, name: str) -> None:
        """The graphs the runner holds now, as bank ``name``."""
        self.banks[name] = self._installed(list(_tensors(self._roots()).values()))
        self.active = name

    def capture(self, name: str) -> Tuple[Optional[str], List[str]]:
        """Capture bank ``name`` with the kernels bound now; the active bank stays
        installed. Returns the error, if the capture failed, and the drift."""
        if name in self.banks:
            raise ValueError(f"bank {name!r} exists")
        active = self.banks[self.active]
        before = _tensors(self._roots())
        fresh = Bank(
            graphs={},
            outputs={},
            capture_inputs=None if active.capture_inputs is None else {},
            metadata=[{bucket: {} for bucket in m} for m in active.metadata],
            glue=None if active.glue is None else {},
            ragged=None if active.ragged is None else {},
            keep=[],
        )
        free = torch.cuda.mem_get_info()[0]
        started = time.perf_counter()
        error = None
        self._install(fresh)
        try:
            with model_capture_mode():
                self.runner.capture()
            torch.cuda.synchronize()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            _end_capture_variant()
            after = _tensors(self._roots())
            self._install(active)
        drift = _moved(before, after)
        if error is None and sorted(map(repr, fresh.graphs)) != sorted(map(repr, active.graphs)):
            error = f"captured {len(fresh.graphs)} graphs where the active bank has {len(active.graphs)}"
        if error is None and not drift:
            fresh.keep = list(after.values())
            fresh.capture_s = time.perf_counter() - started
            fresh.memory_gb = (free - torch.cuda.mem_get_info()[0]) / 2**30
            self.banks[name] = fresh
        else:
            _reset(fresh)
        return error, drift

    def activate(self, name: str) -> None:
        self._install(self.banks[name])
        runner = self.runner
        early = getattr(runner, "_early_verify_metadata", None)
        if early is not None:
            early.discard()
        if getattr(runner, "_staged_inputs", None) is not None:
            # The next load copies its inputs again rather than trust a staging.
            runner._staged_inputs = None
        self.active = name

    def discard(self, name: str) -> None:
        if name == self.active:
            raise ValueError(f"bank {name!r} is active")
        _reset(self.banks.pop(name))

    def drop(self) -> None:
        """Release every bank but the active one, whose graphs the runner keeps."""
        for name in [n for n in self.banks if n != self.active]:
            self.discard(name)

    def fingerprints(self, name: str) -> Dict[str, Optional[tuple]]:
        """Replay each of bank ``name``'s graphs on what the last capture left in
        the static buffers. Only before serving: a replay writes KV through them."""
        previous = self.active
        self.activate(name)
        try:
            runner, backend = self.runner, self.runner.backend
            model_context = getattr(runner, "model_context", None)
            out = {}
            with (
                model_context() if model_context is not None else contextlib.nullcontext(),
                backend.replay_session(),
            ):
                for key in sorted(self.banks[name].graphs, key=repr):
                    out[repr(key)] = _fingerprint(backend.replay(key, None))
            torch.cuda.synchronize()
            return out
        finally:
            self.activate(previous)

    def describe(self) -> dict:
        return {
            "active": self.active,
            "banks": {
                name: {
                    "graphs": len(bank.graphs),
                    "capture_s": None if bank.capture_s is None else round(bank.capture_s, 2),
                    "memory_gb": None if bank.memory_gb is None else round(bank.memory_gb, 2),
                }
                for name, bank in self.banks.items()
            },
        }
