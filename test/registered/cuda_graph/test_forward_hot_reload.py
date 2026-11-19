from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=150, stage="extra-a", runner_config="1-gpu-small")
register_amd_ci(est_time=180, suite="extra-a-test-1-gpu-small-amd")

import hashlib
import tempfile
import time
import unittest
from pathlib import Path

import sglang as sgl
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

PROMPTS = [
    "The capital of France is",
    "Write the numbers from one to ten in words:",
    "A haiku about the ocean:",
]
# Short greedy outputs: a longer continuation can reach a near-tie that flips
# between two identical runs, with or without a swap.
SAMPLING = {"temperature": 0, "max_new_tokens": 8}

# Logs whether each call is being captured into a CUDA graph or runs eagerly.
TRACED = """
import functools

import torch

LOG = {log!r}


def make_forward(original):
    @functools.wraps(original)
    def forward(*args, **kwargs):
        with open(LOG, "a") as f:
            f.write("capture\\n" if torch.cuda.is_current_stream_capturing() else "eager\\n")
        return original(*args, **kwargs)

    return forward
"""


class TestForwardHotReload(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.log = Path(cls.dir.name) / "calls.log"
        cls.candidate = Path(cls.dir.name) / "traced.py"
        cls.candidate.write_text(TRACED.format(log=str(cls.log)))
        started = time.perf_counter()
        cls.engine = sgl.Engine(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            enable_forward_hot_reload=True,
            disable_prefill_cuda_graph=True,
            cuda_graph_max_bs_decode=4,
        )
        cls.startup_s = time.perf_counter() - started

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()
        cls.dir.cleanup()

    def generate(self):
        # A swap flushes the KV cache; prefix-cache hits would change the
        # prefill path and, with it, greedy tokens.
        self.engine.flush_cache()
        outputs = self.engine.generate(PROMPTS, SAMPLING)
        return [out["text"] for out in outputs]

    def calls(self):
        if not self.log.exists():
            return {"capture": 0, "eager": 0}
        lines = self.log.read_text().split()
        return {kind: lines.count(kind) for kind in ("capture", "eager")}

    def test_a_swapped_forward_is_captured_and_serves_the_same_tokens(self):
        baseline = self.generate()
        self.assertEqual(self.calls(), {"capture": 0, "eager": 0})

        started = time.perf_counter()
        out = self.engine.hot_reload_forward(str(self.candidate))
        switch_s = time.perf_counter() - started
        digest = hashlib.sha256(self.candidate.read_bytes()).hexdigest()
        self.assertEqual(out["active"], digest)
        after_reload = self.calls()
        self.assertGreater(after_reload["capture"], 0)

        self.assertEqual(self.generate(), baseline)
        after_generate = self.calls()
        self.assertGreater(after_generate["eager"], after_reload["eager"])

        back = self.engine.restore_forward()
        self.assertEqual(back["active"], "original")
        restored = self.calls()
        self.assertEqual(self.generate(), baseline)
        self.assertEqual(self.calls(), restored)
        print(
            f"startup {self.startup_s:.1f} s, switch {switch_s:.2f} s "
            f"(recapture {out['capture_s']:.2f} s)"
        )

    def test_a_file_that_does_not_match_its_digest_is_refused(self):
        baseline = self.generate()
        with self.assertRaisesRegex(AssertionError, "does not match sha256"):
            self.engine.hot_reload_forward(str(self.candidate), sha256="0" * 64)
        self.assertEqual(self.generate(), baseline)


if __name__ == "__main__":
    unittest.main()
