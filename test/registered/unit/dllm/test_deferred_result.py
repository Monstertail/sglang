"""Stage-1 primitives only: these tests do not enable a serving overlap loop."""

import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.algorithm.joint_threshold import (
    JointThreshold,
    joint_threshold_update_step_vectorized,
)
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.result import DllmDeferredResult
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _CopyDone:
    def __init__(self, ready=True):
        self.ready = ready

    def query(self):
        return self.ready


class _Runner:
    def __init__(self, logits):
        self.logits = logits
        self.calls = 0

    def forward(self, batch, pp_proxy_tensors=None):
        self.calls += 1
        return SimpleNamespace(
            logits_output=SimpleNamespace(full_logits=self.logits), can_run_graph=True
        )


def _algorithm(*, vectorized=True, fdfo=True, **overrides):
    return JointThreshold(
        DllmConfig(
            algorithm="JointThreshold",
            algorithm_config={"vectorized_decoding": vectorized, **overrides},
            block_size=4,
            mask_id=0,
            max_running_requests=32,
            first_done_first_out_mode=fdfo,
        )
    )


def _batch(ids):
    return SimpleNamespace(batch_size=ids.shape[0], input_ids=ids.flatten().clone())


def _legacy_vectorized_step(algorithm, batch, logits, states):
    """Pre-refactor FDFO wrapper, using the unchanged production update kernel."""
    device = batch.input_ids.device
    prompt_masks = torch.stack([state["prompt_mask"] for state in states])
    finished = torch.tensor(
        [state["finished"] for state in states], dtype=torch.bool, device=device
    )
    post_edit_steps = torch.tensor(
        [state["post_edit_steps"] for state in states],
        dtype=torch.int32,
        device=device,
    )
    joint_threshold_update_step_vectorized(
        batch.input_ids,
        logits,
        prompt_masks,
        finished,
        post_edit_steps,
        algorithm.mask_id,
        algorithm.block_size,
        algorithm.threshold,
        algorithm.edit_threshold,
        algorithm.max_post_edit_steps,
        algorithm.penalty_lambda,
    )
    done = finished.tolist()
    steps = post_edit_steps.tolist()
    for i, state in enumerate(states):
        state["finished"] = done[i]
        state["post_edit_steps"] = steps[i]
    return done


class TestDeferredDllmResult(unittest.TestCase):
    def assert_states_equal(self, expected, actual):
        self.assertEqual(len(expected), len(actual))
        for left, right in zip(expected, actual):
            if left is None:
                self.assertIsNone(right)
                continue
            self.assertEqual(left["finished"], right["finished"])
            self.assertEqual(left["post_edit_steps"], right["post_edit_steps"])
            torch.testing.assert_close(left["prompt_mask"], right["prompt_mask"])

    def test_sync_wrapper_matches_pre_refactor_vectorized_step(self):
        for bs in (1, 8, 32):
            for dtype in (torch.float32, torch.bfloat16):
                for threshold, budget, penalty in ((0.5, 16, 0), (1.0, 0, 0.2)):
                    with self.subTest(bs=bs, dtype=dtype, threshold=threshold):
                        algorithm = _algorithm(
                            threshold=threshold,
                            max_post_edit_steps=budget,
                            penalty_lambda=penalty,
                        )
                        ids = torch.zeros((bs, 4), dtype=torch.long)
                        ids[:, 0] = 3
                        if bs > 1:
                            ids[-1] = torch.tensor([1, 2, 3, 4])
                        old, new = _batch(ids), _batch(ids)
                        old_states = algorithm.init_step_state(old)
                        new_states = deepcopy(old_states)
                        generator = torch.Generator().manual_seed(42)
                        for _ in range(3):
                            logits = torch.randn(
                                bs * 4, 19, generator=generator, dtype=dtype
                            )
                            done = _legacy_vectorized_step(
                                algorithm, old, logits.clone(), old_states
                            )
                            self.assertEqual(
                                done, algorithm.step(new, logits.clone(), new_states)
                            )
                            torch.testing.assert_close(old.input_ids, new.input_ids)
                            self.assert_states_equal(old_states, new_states)

    def test_deferred_run_has_no_host_scalar_reads(self):
        algorithm = _algorithm()
        batch = _batch(torch.tensor([[3, 0, 0, 0], [1, 2, 3, 4]]))
        runner = _Runner(torch.randn(8, 19))
        with (
            patch.object(torch.Tensor, "tolist", side_effect=AssertionError("tolist")),
            patch.object(torch.Tensor, "item", side_effect=AssertionError("item")),
            patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("bool")),
        ):
            _, result, can_run_graph = algorithm.run_deferred(runner, batch)
        self.assertEqual(runner.calls, 1)
        self.assertTrue(can_run_graph)
        self.assertEqual(result.block_ids.shape, (2, 4))

    def test_state_pack_happens_before_forward(self):
        algorithm = _algorithm()
        batch = _batch(torch.zeros((1, 4), dtype=torch.long))
        runner = _Runner(torch.randn(4, 19))
        calls = []
        prepare = algorithm.prepare_step
        forward = runner.forward

        def record_prepare(*args):
            calls.append("prepare")
            return prepare(*args)

        def record_forward(*args, **kwargs):
            calls.append("forward")
            return forward(*args, **kwargs)

        with (
            patch.object(algorithm, "prepare_step", side_effect=record_prepare),
            patch.object(runner, "forward", side_effect=record_forward),
        ):
            algorithm.run_deferred(runner, batch)
        self.assertEqual(calls, ["prepare", "forward"])

    def test_result_is_owned_and_materialization_is_event_gated(self):
        algorithm = _algorithm(max_post_edit_steps=0)
        batch = _batch(torch.tensor([[1, 2, 3, 4], [3, 0, 0, 0]]))
        states = algorithm.init_step_state(batch)
        before = deepcopy(states)
        _, result, _ = algorithm.run_deferred(
            _Runner(torch.randn(8, 19)), batch, states
        )
        expected_ids = batch.input_ids.view(2, 4).tolist()
        self.assertIs(result.extra_keep_alive_refs[0], batch)
        self.assert_states_equal(before, states)
        batch.input_ids.fill_(17)
        copies = []

        def copy_tensor(tensor):
            copies.append(tensor.shape)
            return tensor.to("cpu").clone()

        result.map_device_tensors(copy_tensor)
        self.assertEqual(copies, [(2, 4), (2,), (2,)])
        with self.assertRaisesRegex(RuntimeError, "already been submitted"):
            result.map_device_tensors(copy_tensor)
        event = _CopyDone(False)
        with self.assertRaisesRegex(RuntimeError, "copy_done"):
            result.materialize(event)
        self.assert_states_equal(before, states)

        event.ready = True
        materialized = result.materialize(event)
        ids, accepted, carried = materialized
        self.assertEqual(result.extra_keep_alive_refs, ())
        self.assertEqual(ids, expected_ids)
        self.assertEqual(accepted, [4, 0])
        self.assertIsNone(carried[0])
        self.assertIs(carried[1], states[1])
        self.assertIs(carried[1]["prompt_mask"], states[1]["prompt_mask"])
        self.assertEqual(states[0]["post_edit_steps"], 1)
        self.assertIs(result.materialize(event), materialized)
        self.assertEqual(states[0]["post_edit_steps"], 1)
        with self.assertRaisesRegex(RuntimeError, "materialized"):
            result.map_device_tensors(copy_tensor)

    def test_mixed_carried_and_fresh_requests_match_sync_after_reordering(self):
        algorithm = _algorithm()
        ids = torch.tensor([[3, 0, 0, 0], [4, 0, 0, 0], [2, 1, 0, 0]])
        states = algorithm.init_step_state(_batch(ids))
        states[0]["post_edit_steps"] = 3
        states[2]["post_edit_steps"] = 7
        carried = [states[2], None, states[0]]
        reordered = ids[[2, 1, 0]]
        sync_batch, async_batch = _batch(reordered), _batch(reordered)
        logits = torch.randn(12, 19, generator=torch.Generator().manual_seed(123))
        sync_runner, async_runner = _Runner(logits.clone()), _Runner(logits.clone())
        sync = algorithm.run(sync_runner, sync_batch, deepcopy(carried))
        _, pending, can_run_graph = algorithm.run_deferred(
            async_runner, async_batch, deepcopy(carried)
        )
        pending.map_device_tensors(lambda tensor: tensor.to("cpu").clone())
        actual_ids, actual_accept, actual_states = pending.materialize(_CopyDone())
        self.assertEqual(actual_ids, sync[1])
        self.assertEqual(actual_accept, sync[2])
        self.assert_states_equal(sync[3], actual_states)
        self.assertEqual(can_run_graph, sync[4])
        self.assertEqual((sync_runner.calls, async_runner.calls), (1, 1))

    def test_exact_ties_and_finished_rows_match_legacy_wrapper(self):
        algorithm = _algorithm(threshold=1.0, max_post_edit_steps=0)
        ids = torch.tensor([[3, 0, 0, 0], [1, 2, 3, 4]])
        old, new = _batch(ids), _batch(ids)
        old_states = algorithm.init_step_state(old)
        old_states[1]["finished"] = True
        new_states = deepcopy(old_states)
        logits = torch.zeros((8, 19))
        done = _legacy_vectorized_step(algorithm, old, logits.clone(), old_states)
        self.assertEqual(done, algorithm.step(new, logits.clone(), new_states))
        torch.testing.assert_close(old.input_ids, new.input_ids)
        self.assert_states_equal(old_states, new_states)

    def test_unsupported_algorithms_fail_before_forward(self):
        batch = _batch(torch.zeros((1, 4), dtype=torch.long))
        for algorithm in (_algorithm(vectorized=False), _algorithm(fdfo=False)):
            runner = _Runner(torch.randn(4, 19))
            with self.assertRaises(ValueError):
                algorithm.run_deferred(runner, batch)
            self.assertEqual(runner.calls, 0)
        runner = _Runner(torch.randn(4, 19))
        algorithm = DllmAlgorithm(
            DllmConfig("unsupported", {}, 4, 0, 1, first_done_first_out_mode=True)
        )
        with self.assertRaises(NotImplementedError):
            algorithm.run_deferred(runner, batch)
        self.assertEqual(runner.calls, 0)

    def test_carried_state_count_is_validated_before_forward(self):
        algorithm = _algorithm()
        batch = _batch(torch.zeros((2, 4), dtype=torch.long))
        runner = _Runner(torch.randn(8, 19))
        for run in (algorithm.run, algorithm.run_deferred):
            with self.assertRaisesRegex(ValueError, "one carried state"):
                run(runner, batch, [None])
        self.assertEqual(runner.calls, 0)

    def test_invalid_result_shape_and_row_count_are_rejected(self):
        state = SimpleNamespace(materialize=lambda: ([True], [{}]))
        with self.assertRaisesRegex(ValueError, "shape"):
            DllmDeferredResult(torch.zeros(4), state)
        result = DllmDeferredResult(torch.zeros((2, 4)), state)
        with self.assertRaisesRegex(ValueError, "one entry"):
            result.materialize(_CopyDone())

    @unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
    def test_cuda_copy_event_and_snapshot(self):
        from sglang.srt.managers.utils import _async_d2h

        algorithm = _algorithm()
        ids = torch.tensor([[3, 0, 0, 0], [1, 2, 3, 4]], device="cuda")
        batch = _batch(ids)
        logits = torch.randn(8, 19, device="cuda")
        expected = algorithm.run(_Runner(logits.clone()), _batch(ids))
        forward_stream, copy_stream = torch.cuda.Stream(), torch.cuda.Stream()
        forward_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(forward_stream):
            _, pending, _ = algorithm.run_deferred(_Runner(logits.clone()), batch)
        copy_stream.wait_stream(forward_stream)
        event = torch.cuda.Event()
        with torch.cuda.stream(copy_stream):
            pending.map_device_tensors(_async_d2h)
            event.record()
        event.synchronize()
        actual_ids, actual_accept, actual_states = pending.materialize(event)
        self.assertEqual(actual_ids, expected[1])
        self.assertEqual(actual_accept, expected[2])
        self.assert_states_equal(expected[3], actual_states)
        self.assertTrue(pending.block_ids.is_pinned())


if __name__ == "__main__":
    unittest.main()
