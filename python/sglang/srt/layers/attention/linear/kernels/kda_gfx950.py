from __future__ import annotations

import torch
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)


class Gfx950KDAKernel(LinearAttnKernelBase):
    """Generated HIP KDA packed-prefill kernel for MI355X."""

    supports_track_state_snapshot: bool = True

    def __init__(self) -> None:
        self._fallback = TritonKDAKernel()

    def decode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Gfx950KDAKernel only supports prefill (extend)")

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        lower_bound: float | None = None,
        is_spec_decode: bool = False,
        beta_is_raw: bool = False,
        return_intermediate_states: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        track_state = kwargs.get("track_state")
        track_chunk_idx = kwargs.get("track_chunk_idx")
        if self._should_fall_back(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            ssm_states=ssm_states,
            lower_bound=lower_bound,
            A_log=A_log,
            dt_bias=dt_bias,
            is_spec_decode=is_spec_decode,
            beta_is_raw=beta_is_raw,
            return_intermediate_states=return_intermediate_states,
            track_state=track_state,
            track_chunk_idx=track_chunk_idx,
            chunk_indices=kwargs.get("chunk_indices"),
        ):
            return self._fallback.extend(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                beta_is_raw=beta_is_raw,
                return_intermediate_states=return_intermediate_states,
                **kwargs,
            )

        from sglang.kernels.ops.attention.linear.kda_gfx950_prefill import (
            kda_prefill_gfx950,
        )

        selected_state = ssm_states[cache_indices].contiguous()
        output, final_state = kda_prefill_gfx950(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g.contiguous(),
            beta.contiguous(),
            A_log.reshape(-1).float().contiguous(),
            dt_bias.reshape(-1).float().contiguous(),
            selected_state,
            query_start_loc.to(torch.int64).contiguous(),
            q.shape[-1] ** -0.5,
            track_state=track_state,
            track_chunk_idx=(
                track_chunk_idx.to(torch.int64).contiguous()
                if track_chunk_idx is not None
                else None
            ),
        )
        ssm_states[cache_indices] = final_state
        return (output, None) if return_intermediate_states else output

    @staticmethod
    def _should_fall_back(
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        ssm_states: torch.Tensor,
        lower_bound: float | None,
        A_log: torch.Tensor | None,
        dt_bias: torch.Tensor | None,
        is_spec_decode: bool,
        beta_is_raw: bool,
        return_intermediate_states: bool,
        track_state: torch.Tensor | None,
        track_chunk_idx: torch.Tensor | None,
        chunk_indices: torch.Tensor | None,
    ) -> bool:
        if is_spec_decode or chunk_indices is not None:
            return True
        if (track_state is None) != (track_chunk_idx is None):
            return True
        if return_intermediate_states and track_state is None:
            return True
        if lower_bound != -5.0 or not beta_is_raw:
            return True
        if A_log is None or dt_bias is None:
            return True
        if (
            q.dtype != torch.bfloat16
            or k.dtype != torch.bfloat16
            or v.dtype != torch.bfloat16
            or g.dtype != torch.bfloat16
            or beta.dtype != torch.bfloat16
            or ssm_states.dtype != torch.float32
        ):
            return True
        if not (q.shape == k.shape == v.shape == g.shape):
            return True
        return (
            q.ndim != 4
            or q.shape[0] != 1
            or q.shape[-1] != 128
            or ssm_states.shape[-2:] != (128, 128)
        )
