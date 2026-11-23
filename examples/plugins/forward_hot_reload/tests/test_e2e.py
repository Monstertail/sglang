"""End to end on one GPU: the plugin installed, a small model, a candidate that
logs whether it runs eagerly or is being captured into a CUDA graph.

Run:  pip install -e examples/plugins/forward_hot_reload
      python -m pytest examples/plugins/forward_hot_reload/tests/test_e2e.py -v -s
"""

import hashlib
import os
import tempfile
import time
import unittest
from pathlib import Path

SAMPLING = {"temperature": 0, "max_new_tokens": 8}
PROMPTS = [
    "The capital of France is",
    "Write the numbers from one to ten in words:",
    "A haiku about the ocean:",
]
MODEL = os.environ.get(
    "SGLANG_FORWARD_HOT_RELOAD_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"
)

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


def _have_sglang_and_a_gpu() -> bool:
    try:
        import torch

        import sglang  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


@unittest.skipUnless(_have_sglang_and_a_gpu(), "needs SGLang and a GPU")
class TestPluginEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Scheduler processes inherit the switch; the entry point reads it there.
        os.environ["SGLANG_FORWARD_HOT_RELOAD"] = "1"
        import sglang as sgl

        cls.dir = tempfile.TemporaryDirectory()
        cls.log = Path(cls.dir.name) / "calls.log"
        cls.candidate = Path(cls.dir.name) / "traced.py"
        cls.candidate.write_text(TRACED.format(log=str(cls.log)))
        started = time.perf_counter()
        cls.engine = sgl.Engine(
            model_path=MODEL,
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
        return [out["text"] for out in self.engine.generate(PROMPTS, SAMPLING)]

    def calls(self):
        if not self.log.exists():
            return {"capture": 0, "eager": 0}
        lines = self.log.read_text().split()
        return {kind: lines.count(kind) for kind in ("capture", "eager")}

    def test_a_swapped_forward_is_captured_and_serves_the_same_tokens(self):
        from sglang_forward_hot_reload import hot_reload_forward, restore_forward

        baseline = self.generate()
        self.assertEqual(self.calls(), {"capture": 0, "eager": 0})

        started = time.perf_counter()
        out = hot_reload_forward(self.engine, str(self.candidate))
        switch_s = time.perf_counter() - started
        digest = hashlib.sha256(self.candidate.read_bytes()).hexdigest()
        self.assertEqual(out["active"], digest)
        after_reload = self.calls()
        self.assertGreater(after_reload["capture"], 0)

        self.assertEqual(self.generate(), baseline)
        self.assertGreater(self.calls()["eager"], after_reload["eager"])

        back = restore_forward(self.engine)
        self.assertEqual(back["active"], "original")
        restored = self.calls()
        self.assertEqual(self.generate(), baseline)
        self.assertEqual(self.calls(), restored)
        print(
            f"startup {self.startup_s:.1f} s, switch {switch_s:.2f} s "
            f"(recapture {out['capture_s']:.2f} s)"
        )

    def test_a_file_that_does_not_match_its_digest_is_refused(self):
        from sglang_forward_hot_reload import hot_reload_forward

        baseline = self.generate()
        with self.assertRaisesRegex(AssertionError, "does not match sha256"):
            hot_reload_forward(self.engine, str(self.candidate), sha256="0" * 64)
        self.assertEqual(self.generate(), baseline)


if __name__ == "__main__":
    unittest.main()
