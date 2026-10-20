# SPDX-License-Identifier: Apache-2.0

import itertools
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.linear.kernels.kda_gfx950 import Gfx950KDAKernel
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=300, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _is_gfx950() -> bool:
    return bool(
        torch.version.hip
        and torch.cuda.is_available()
        and torch.cuda.get_device_properties().gcnArchName.split(":", 1)[0] == "gfx950"
    )


def _inputs(lengths: tuple[int, ...] = (193, 137)) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(42)
    total_tokens = sum(lengths)
    heads = 4
    shape = (1, total_tokens, heads, 128)

    def randn(shape, dtype, scale):
        return (torch.randn(shape, generator=generator, device="cuda") * scale).to(
            dtype
        )

    return {
        "q": randn(shape, torch.bfloat16, 0.5),
        "k": randn(shape, torch.bfloat16, 0.5),
        "v": randn(shape, torch.bfloat16, 0.5),
        "g": randn(shape, torch.bfloat16, 0.5),
        "beta": randn(shape[:3], torch.bfloat16, 0.5),
        "A_log": torch.randn(
            heads, generator=generator, device="cuda", dtype=torch.float32
        ),
        "dt_bias": torch.randn(
            heads * 128, generator=generator, device="cuda", dtype=torch.float32
        ),
        "ssm_states": randn((7, heads, 128, 128), torch.float32, 0.1),
        "cache_indices": torch.tensor(
            (1, 4)[: len(lengths)], device="cuda", dtype=torch.int32
        ),
        "query_start_loc": torch.tensor(
            (0, *itertools.accumulate(lengths)), device="cuda", dtype=torch.int32
        ),
    }


def _clone(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.clone() for name, value in inputs.items()}


def _run(kernel, inputs, *, unwrap=True, **kwargs):
    result = kernel.extend(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        ssm_states=inputs["ssm_states"],
        cache_indices=inputs["cache_indices"],
        query_start_loc=inputs["query_start_loc"],
        lower_bound=-5.0,
        beta_is_raw=True,
        return_intermediate_states=bool(kwargs),
        **kwargs,
    )
    return result[0] if unwrap and isinstance(result, tuple) else result


@unittest.skipUnless(_is_gfx950(), "gfx950 KDA prefill requires MI35x")
class TestGfx950KDAPrefill(CustomTestCase):
    @torch.inference_mode()
    def test_untracked_extend_returns_tensor(self):
        inputs = _inputs()
        candidate = Gfx950KDAKernel()
        with patch.object(
            candidate._fallback,
            "extend",
            side_effect=AssertionError("gfx950 KDA unexpectedly fell back to Triton"),
        ):
            result = _run(candidate, inputs, unwrap=False)
        self.assertIsInstance(result, torch.Tensor)

    @torch.inference_mode()
    def test_aligned_track_extend_skips_snapshot(self):
        inputs = _inputs()
        expected_inputs = _clone(inputs)
        actual_inputs = _clone(inputs)
        aligned = torch.empty(0, device="cuda", dtype=torch.int64)
        expected = _run(TritonKDAKernel(), expected_inputs, track_ssm_h_src=aligned)
        candidate = Gfx950KDAKernel()
        with patch.object(
            candidate._fallback,
            "extend",
            side_effect=AssertionError("gfx950 KDA unexpectedly fell back to Triton"),
        ):
            actual, h = _run(
                candidate, actual_inputs, unwrap=False, track_ssm_h_src=aligned
            )
        torch.cuda.synchronize()

        self.assertIsNone(h)
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=2e-2, atol=3e-2
        )
        torch.testing.assert_close(
            actual_inputs["ssm_states"],
            expected_inputs["ssm_states"],
            rtol=2e-2,
            atol=3e-2,
        )

    @torch.inference_mode()
    def test_varlen_state_and_track_snapshot_match_triton(self):
        self._check_track_snapshot((193, 137), (2, 1))

    @torch.inference_mode()
    def test_long_sequence_groups_match_triton(self):
        # one sequence long enough for the sequence-group recurrence; chunk 3 is
        # tracked in the first group, chunk 40 in a later one
        for track_chunk in (3, 40):
            self._check_track_snapshot((4000,), (track_chunk,))

    def _check_track_snapshot(self, lengths, track_chunks):
        inputs = _inputs(lengths)
        expected_inputs = _clone(inputs)
        actual_inputs = _clone(inputs)
        expected_track = torch.empty(
            len(lengths), 4, 128, 128, device="cuda", dtype=torch.float32
        )
        actual_track = torch.empty_like(expected_track)
        track_chunk_idx = torch.tensor(track_chunks, device="cuda", dtype=torch.int64)

        expected = _run(
            TritonKDAKernel(),
            expected_inputs,
            track_state=expected_track,
            track_chunk_idx=track_chunk_idx,
        )
        candidate = Gfx950KDAKernel()
        with patch.object(
            candidate._fallback,
            "extend",
            side_effect=AssertionError("gfx950 KDA unexpectedly fell back to Triton"),
        ):
            actual = _run(
                candidate,
                actual_inputs,
                track_state=actual_track,
                track_chunk_idx=track_chunk_idx,
            )
        torch.cuda.synchronize()

        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=2e-2, atol=3e-2
        )
        torch.testing.assert_close(
            actual_inputs["ssm_states"],
            expected_inputs["ssm_states"],
            rtol=2e-2,
            atol=3e-2,
        )
        torch.testing.assert_close(actual_track, expected_track, rtol=2e-2, atol=3e-2)
        selected = torch.zeros(7, dtype=torch.bool, device="cuda")
        selected[inputs["cache_indices"].long()] = True
        self.assertTrue(
            torch.equal(
                actual_inputs["ssm_states"][~selected], inputs["ssm_states"][~selected]
            )
        )


if __name__ == "__main__":
    unittest.main()
