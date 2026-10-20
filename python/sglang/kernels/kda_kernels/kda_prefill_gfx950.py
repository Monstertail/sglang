# SPDX-License-Identifier: Apache-2.0
"""HIP Kimi Delta Attention chunked-prefill kernel for gfx950 (MI350X / MI355X).

The extension is JIT-compiled with ``torch.utils.cpp_extension`` on first use
and cached under ``TORCH_EXTENSIONS_DIR``.
"""

from __future__ import annotations

import os

import torch

from sglang.kernels.kda_kernels import _cuda_source

_EXT_NAME = "kda_prefill_gfx950"
_ext = None


def load_ext():
    """JIT-load the HIP extension (cached after first call)."""
    global _ext
    if _ext is None:
        from sglang.srt.utils.cpp_extension_loader import (
            load_extension_with_recovery,
        )

        # torch builds for the archs in PYTORCH_ROCM_ARCH (its own defaults when
        # unset), and the kernel's MFMA builtins only exist on gfx950.
        previous_arch = os.environ.get("PYTORCH_ROCM_ARCH")
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
        try:
            _ext = load_extension_with_recovery(
                name=_EXT_NAME,
                sources=[_cuda_source("attention/kda_prefill_gfx950.cu")],
                extra_cuda_cflags=["-O3", "-mllvm", "-amdgpu-mfma-vgpr-form"],
                extra_cflags=["-O3"],
            )
        finally:
            if previous_arch is None:
                os.environ.pop("PYTORCH_ROCM_ARCH", None)
            else:
                os.environ["PYTORCH_ROCM_ARCH"] = previous_arch
    return _ext


def kda_prefill_gfx950(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    track_state: torch.Tensor | None = None,
    track_chunk_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Returns the output; ssm_states[cache_indices] is updated in place."""
    if track_state is None:
        track_state = torch.empty(0, dtype=torch.float32, device=q.device)
    if track_chunk_idx is None:
        track_chunk_idx = torch.empty(0, dtype=torch.int64, device=q.device)
    return load_ext().kda_prefill_gfx950(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        ssm_states,
        cache_indices,
        cu_seqlens,
        track_state,
        track_chunk_idx,
        scale,
    )


__all__ = ["kda_prefill_gfx950"]
