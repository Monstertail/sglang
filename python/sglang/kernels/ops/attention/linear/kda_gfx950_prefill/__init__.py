# SPDX-License-Identifier: Apache-2.0
"""JIT loader for the generated gfx950 KDA packed-prefill kernel."""

from __future__ import annotations

import os

import torch

_EXT_NAME = "kda_prefill_gfx950"
_ext = None


def load_ext():
    """Build and cache the gfx950 HIP extension."""
    global _ext
    if _ext is None:
        from sglang.srt.utils.cpp_extension_loader import (
            load_extension_with_recovery,
        )

        src = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../../../jit/csrc/attention/kda_prefill_gfx950.cu",
        )
        previous_arch = os.environ.get("PYTORCH_ROCM_ARCH")
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
        try:
            _ext = load_extension_with_recovery(
                name=_EXT_NAME,
                sources=[src],
                extra_cuda_cflags=[
                    "-O3",
                    "--offload-arch=gfx950",
                    "-DKDA_VCVT=1",
                    "-DKDA_ROLL=1",
                    "-mllvm",
                    "-amdgpu-mfma-vgpr-form",
                ],
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
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    track_state: torch.Tensor | None = None,
    track_chunk_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        initial_state,
        cu_seqlens,
        track_state,
        track_chunk_idx,
        scale,
    )


__all__ = ["kda_prefill_gfx950"]
