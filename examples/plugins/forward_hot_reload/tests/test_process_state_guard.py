"""Unit tests for ProcessStateGuard. torch is a stand-in; no GPU.

Run:  python -m pytest examples/plugins/forward_hot_reload/tests -v
"""

import json
import types
import unittest

from sglang_forward_hot_reload.process_state_guard import ProcessStateGuard


def fake_torch():
    state = dict(
        dtype="torch.bfloat16", precision="highest", deterministic=False, stream=7
    )
    matmul = types.SimpleNamespace(
        allow_tf32=False,
        allow_fp16_reduced_precision_reduction=True,
        allow_bf16_reduced_precision_reduction=True,
    )
    cudnn = types.SimpleNamespace(allow_tf32=True, deterministic=False, benchmark=False)
    torch_ = types.SimpleNamespace(
        backends=types.SimpleNamespace(
            cuda=types.SimpleNamespace(matmul=matmul), cudnn=cudnn
        ),
        get_default_dtype=lambda: state["dtype"],
        get_float32_matmul_precision=lambda: state["precision"],
        are_deterministic_algorithms_enabled=lambda: state["deterministic"],
        is_grad_enabled=lambda: False,
        is_inference_mode_enabled=lambda: False,
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            current_stream=lambda: types.SimpleNamespace(cuda_stream=state["stream"]),
        ),
    )
    torch_.state = state
    return torch_


class TestProcessStateGuard(unittest.TestCase):
    def guard(self, **env):
        self.environ = {"HIP_VISIBLE_DEVICES": "0,1,2,3", **env}
        self.path = ["/a", "/b"]
        self.torch = fake_torch()
        return ProcessStateGuard(self.torch, self.environ, self.path)

    def test_a_fingerprint_names_variables_and_holds_no_value(self):
        guard = self.guard(SECRET_TOKEN="hunter2-hunter2")
        text = json.dumps(guard.now())
        self.assertNotIn("hunter2", text)
        self.assertIn("SECRET_TOKEN", text)
        self.assertEqual(guard.now()["flags"]["matmul_precision"], "highest")

    def test_nothing_is_compared_before_the_reference(self):
        guard = self.guard()
        self.torch.backends.cuda.matmul.allow_tf32 = True
        self.assertIsNone(guard.moved())
        self.assertEqual(guard.verify(), "ok")

    def test_a_moved_flag_is_fatal_and_named(self):
        guard = self.guard()
        guard.reference()
        self.assertEqual(guard.verify(), "ok")
        self.torch.backends.cuda.matmul.allow_tf32 = True
        with self.assertRaisesRegex(
            RuntimeError, "process state drifted: tf32_matmul False -> True"
        ):
            guard.verify()

    def test_dtype_stream_and_sys_path_count(self):
        guard = self.guard()
        guard.reference()
        self.torch.state.update(dtype="torch.float16", stream=9)
        self.path.append("/candidate")
        with self.assertRaises(RuntimeError) as caught:
            guard.verify()
        for part in (
            "default_dtype torch.bfloat16 -> torch.float16",
            "stream 7 -> 9",
            "sys.path changed",
        ):
            self.assertIn(part, str(caught.exception))

    def test_a_moved_environment_variable_is_reported_not_fatal(self):
        guard = self.guard()
        guard.reference()
        self.environ["SOME_LIBRARY_CACHE"] = "/x"
        self.environ["HIP_VISIBLE_DEVICES"] = "3"
        self.assertEqual(guard.verify(), "ok")
        self.assertEqual(
            guard.moved()["env"],
            ["HIP_VISIBLE_DEVICES changed", "SOME_LIBRARY_CACHE added"],
        )
        self.assertNotIn("0,1,2,3", json.dumps(guard.moved()))

    def test_the_reference_is_taken_once(self):
        guard = self.guard()
        guard.reference()
        self.torch.state["precision"] = "high"
        guard.reference()
        self.assertEqual(guard.moved()["flags"], ["matmul_precision highest -> high"])

    def test_a_flag_this_torch_lacks_reads_the_same_both_times(self):
        guard = self.guard()
        del self.torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        guard.reference()
        self.assertTrue(
            guard.now()["flags"]["bf16_reduced_precision_reduction"].startswith(
                "unreadable"
            )
        )
        self.assertEqual(guard.verify(), "ok")


if __name__ == "__main__":
    unittest.main()
