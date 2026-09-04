from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Protocol, Tuple

if TYPE_CHECKING:
    import torch


class DllmStepState(Protocol):
    """Algorithm-owned state for a device step and its later host commit."""

    def map_device_tensors(self, copy_tensor: Callable[[torch.Tensor], torch.Tensor]):
        """Copy only the tensors needed by ``materialize`` to host memory."""
        ...

    def materialize(self) -> Tuple[List[bool], List[Any]]:
        """Read ready host tensors, returning done flags and per-request states."""
        ...


class DllmCopyDone(Protocol):
    def query(self) -> bool: ...


@dataclass
class DllmDeferredResult:
    """An owned block snapshot plus opaque algorithm state, not an AR token relay.

    The caller orders the copy stream after the device update, maps tensors with
    the shared ``_async_d2h`` helper, and records ``copy_done`` after all copies.
    Keep this holder alive until that event completes. ``materialize`` never
    waits implicitly: the scheduler owns the result-consumption boundary.
    """

    block_ids: torch.Tensor
    step_state: DllmStepState
    extra_keep_alive_refs: Tuple[Any, ...] = field(default=(), repr=False)
    _copy_started: bool = field(default=False, init=False, repr=False)
    _materialized: Optional[Tuple[List[List[int]], List[int], List[Any]]] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self):
        if self.block_ids.ndim != 2:
            raise ValueError("Deferred dLLM block IDs must have shape [batch, block]")

    def map_device_tensors(self, copy_tensor: Callable[[torch.Tensor], torch.Tensor]):
        if self._materialized is not None:
            raise RuntimeError("Cannot copy a materialized dLLM result")
        if self._copy_started:
            raise RuntimeError("dLLM result copies have already been submitted")
        self._copy_started = True
        self.block_ids = copy_tensor(self.block_ids)
        self.step_state.map_device_tensors(copy_tensor)

    def materialize(
        self, copy_done: DllmCopyDone
    ) -> Tuple[List[List[int]], List[int], List[Any]]:
        if self._materialized is not None:
            return self._materialized
        if not copy_done.query():
            raise RuntimeError("dLLM result is not ready; wait for copy_done first")
        if self.block_ids.device.type != "cpu":
            raise RuntimeError("dLLM block IDs must be copied to CPU before commit")

        done, states = self.step_state.materialize()
        if len(done) != self.block_ids.shape[0] or len(states) != len(done):
            raise ValueError("Deferred dLLM state must have one entry per block")
        block_size = self.block_ids.shape[1]
        self._materialized = (
            self.block_ids.tolist(),
            [block_size if finished else 0 for finished in done],
            [None if finished else state for finished, state in zip(done, states)],
        )
        # The event covers forward/update and all result copies. Prompt-mask
        # tensors needed by later rounds remain owned by the carried states.
        self.extra_keep_alive_refs = ()
        return self._materialized
