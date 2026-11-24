"""Unit tests for prebuilt kernel binaries: the manifest, what it catches, and the
serving guard. CPU only; the caches are files written by hand.

Run:  python -m pytest test/registered/unit/managers/test_kernel_binaries.py -v
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from sglang.srt.managers.scheduler_components import kernel_binaries as kb
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

TOOLCHAIN = {"flydsl": "0.4.2", "flydsl_key": "abc", "triton": "3.4.0", "arch": "gfx950"}
FILES = {
    "flydsl/stage1_k1/aaaa.pkl": b"stage1",
    "triton/H1/_rmsnorm_kernel.hsaco": b"hsaco",
    "triton/H1/__grp___rmsnorm_kernel.json": b"{}",
}


class TestKernelBinaries(CustomTestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name) / "bin"
        for name, data in FILES.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        # Triton's staging file is not an entry.
        (self.root / "triton/H1/tmp.123.json").write_bytes(b"partial")
        self.spec = Path(self._dir.name) / "profiles.json"
        self.spec.write_text(json.dumps({"binaries": {"dir": str(self.root)}}))
        self._env = patch.dict(os.environ, {}, clear=False)
        self._env.start()
        for variable in (*kb.CACHES.values(), kb.RUN_ONLY):
            os.environ.pop(variable, None)
        self._toolchain = patch.object(kb, "toolchain", return_value=dict(TOOLCHAIN))
        self._toolchain.start()

    def tearDown(self):
        self._toolchain.stop()
        self._env.stop()
        self._dir.cleanup()

    def built(self, **spec):
        kb.write_manifest(self.root, profiles=["tuned"])
        kb.configure(str(self.spec))
        return kb.Binaries({"dir": str(self.root), **spec})

    def test_manifest_lists_every_kernel_file(self):
        manifest = kb.write_manifest(self.root, profiles=["tuned"])
        self.assertEqual(sorted(manifest["entries"]), sorted(FILES))
        self.assertEqual(manifest["toolchain"], TOOLCHAIN)
        self.assertEqual(
            sorted({kb.kernel(n) for n in manifest["entries"]}),
            ["flydsl:stage1", "triton:_rmsnorm_kernel"],
        )

    def test_configure_points_both_compilers_into_the_set(self):
        self.assertEqual(kb.configure(str(self.spec)), str(self.root.resolve()))
        self.assertEqual(os.environ["FLYDSL_RUNTIME_CACHE_DIR"], str(self.root.resolve() / "flydsl"))
        self.assertEqual(os.environ["TRITON_CACHE_DIR"], str(self.root.resolve() / "triton"))
        # aiter's shipped cache and SGLang's own default give way; a directory set
        # by hand does not.
        with patch.object(kb, "shipped_flydsl_cache", return_value="/opt/aiter/jit/flydsl_cache"):
            os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = "/opt/aiter/jit/flydsl_cache"
            kb.configure(str(self.spec))
        self.assertEqual(os.environ["FLYDSL_RUNTIME_CACHE_DIR"], str(self.root.resolve() / "flydsl"))
        os.environ["TRITON_CACHE_DIR"] = kb._default("TRITON_CACHE_DIR")
        kb.configure(str(self.spec))
        self.assertEqual(os.environ["TRITON_CACHE_DIR"], str(self.root.resolve() / "triton"))
        os.environ["TRITON_CACHE_DIR"] = "/elsewhere"
        with self.assertRaisesRegex(ValueError, "TRITON_CACHE_DIR=/elsewhere"):
            kb.configure(str(self.spec))
        self.assertIsNone(kb.configure(None))

    def test_a_clean_set_has_no_problems(self):
        binaries = self.built()
        self.assertEqual(binaries.check(), [])
        self.assertEqual(binaries.compiled_at_startup, [])
        described = binaries.describe()
        self.assertEqual((described["entries"], described["serving"]), (3, "warn"))

    def test_what_startup_and_serving_compiled(self):
        binaries = self.built()
        (self.root / "flydsl/stage2_k9").mkdir()
        (self.root / "flydsl/stage2_k9/bbbb.pkl").write_bytes(b"new")
        binaries.check()
        self.assertEqual(binaries.compiled_at_startup, ["flydsl:stage2"])
        (self.root / "triton/H2").mkdir()
        (self.root / "triton/H2/_late_kernel.json").write_bytes(b"{}")
        self.assertEqual(binaries.compiled_while_serving(), ["triton:_late_kernel"])

    def test_changed_missing_and_foreign_entries(self):
        binaries = self.built()
        (self.root / "flydsl/stage1_k1/aaaa.pkl").write_bytes(b"STAGE1")
        (self.root / "triton/H1/_rmsnorm_kernel.hsaco").unlink()
        problems = binaries.check()
        self.assertIn("flydsl/stage1_k1/aaaa.pkl is not the one built", problems)
        self.assertIn("triton/H1/_rmsnorm_kernel.hsaco is missing", problems)
        # Sizes only: the same-size rewrite goes unseen.
        self.assertNotIn(
            "flydsl/stage1_k1/aaaa.pkl is not the one built",
            kb.Binaries({"dir": str(self.root), "verify": False}).check(),
        )

    def test_another_toolchain_or_cache_dir_is_a_problem(self):
        binaries = self.built()
        with patch.object(kb, "toolchain", return_value={**TOOLCHAIN, "flydsl_key": "def"}):
            self.assertIn("built with flydsl_key abc, this process has def", binaries.check())
        os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = "/tmp/other"
        self.assertTrue(any(p.startswith("FLYDSL_RUNTIME_CACHE_DIR is") for p in binaries.check()))

    def test_strict_closes_flydsl_to_compiles(self):
        binaries = self.built(serving="strict")
        binaries.check()
        binaries.guard()
        self.assertEqual(os.environ[kb.RUN_ONLY], "1")
        warn = kb.Binaries({"dir": str(self.root)})
        os.environ.pop(kb.RUN_ONLY)
        warn.guard()
        self.assertNotIn(kb.RUN_ONLY, os.environ)
        with self.assertRaisesRegex(ValueError, "serving is one of"):
            kb.Binaries({"dir": str(self.root), "serving": "never"})
