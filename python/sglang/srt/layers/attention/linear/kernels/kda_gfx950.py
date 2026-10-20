from __future__ import annotations

import torch

from sglang.kernels.ops.attention import kda_prefill_gfx950
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)


class Gfx950KDAKernel(LinearAttnKernelBase):
    """HIP KDA chunked-prefill kernel for gfx950 (MI350X / MI355X)."""

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
        track_ssm_h_src = kwargs.get("track_ssm_h_src")
        supported = (
            not is_spec_decode
            and kwargs.get("chunk_indices") is None
            and (track_state is None) == (track_chunk_idx is None)
            # Chunk-aligned tracking reads only the final state, never the per-chunk h.
            and not (
                return_intermediate_states
                and track_state is None
                and (track_ssm_h_src is None or track_ssm_h_src.numel() > 0)
            )
            and lower_bound == -5.0
            and beta_is_raw
            and A_log is not None
            and dt_bias is not None
            and all(x.dtype == torch.bfloat16 for x in (q, k, v, g, beta))
            and ssm_states.dtype == torch.float32
            and ssm_states.is_contiguous()
            and q.shape == k.shape == v.shape == g.shape
            and q.ndim == 4
            and q.shape[0] == 1
            and q.shape[-1] == 128
            and ssm_states.shape[-2:] == (128, 128)
        )
        if not supported:
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

        output = kda_prefill_gfx950(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g.contiguous(),
            beta.contiguous(),
            A_log.reshape(-1).float().contiguous(),
            dt_bias.reshape(-1).float().contiguous(),
            ssm_states,
            cache_indices.to(torch.int32).contiguous(),
            query_start_loc.to(torch.int32).contiguous(),
            q.shape[-1] ** -0.5,
            track_state=track_state,
            track_chunk_idx=(
                track_chunk_idx.to(torch.int64).contiguous()
                if track_chunk_idx is not None
                else None
            ),
        )
        return (output, None) if return_intermediate_states else output
