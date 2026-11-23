"""The entry point and the Engine-side helpers, against stand-ins for SGLang's
Scheduler module and the engine. Needs torch only.

Run:  python -m pytest examples/plugins/forward_hot_reload/tests -v
"""

import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import sglang_forward_hot_reload as plugin
from sglang_forward_hot_reload import forward_hot_reload as fhr


def scheduler_modules(**attributes):
    scheduler = type("Scheduler", (), attributes)
    names = ("sglang", "sglang.srt", "sglang.srt.managers")
    modules = {name: types.ModuleType(name) for name in names}
    modules["sglang.srt.managers.scheduler"] = types.SimpleNamespace(
        Scheduler=scheduler
    )
    return modules, scheduler


class FakeEngine:
    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    def collective_rpc(self, method, **kwargs):
        if self.fail:
            raise AssertionError(self.fail)
        status = Path(kwargs.pop("status_dir"))
        self.calls.append((method, kwargs, sorted(os.listdir(status))))
        for rank in (0, 1):
            (status / f"rank-{rank}.json").write_text(
                json.dumps({"rank": rank, "method": method, **kwargs})
            )


ON = {plugin.ENABLE_ENV: "1"}


class TestRegister(unittest.TestCase):
    def test_without_the_switch_nothing_is_added(self):
        modules, scheduler = scheduler_modules()
        with patch.dict(sys.modules, modules), patch.dict(os.environ):
            os.environ.pop(plugin.ENABLE_ENV, None)
            plugin.register()
        for name in plugin.RPCS:
            self.assertFalse(hasattr(scheduler, name))

    def test_with_the_switch_both_rpcs_become_scheduler_methods(self):
        modules, scheduler = scheduler_modules()
        with patch.dict(sys.modules, modules), patch.dict(os.environ, ON):
            plugin.register()
            plugin.register()  # every process runs it; once more changes nothing
        self.assertIs(scheduler.hot_reload_forward, fhr.hot_reload_forward)
        self.assertIs(scheduler.restore_forward, fhr.restore_forward)

    def test_a_scheduler_attribute_of_the_same_name_is_not_replaced(self):
        mine = object()
        modules, scheduler = scheduler_modules(restore_forward=mine)
        with patch.dict(sys.modules, modules), patch.dict(os.environ, ON):
            with self.assertRaisesRegex(RuntimeError, "already has 'restore_forward'"):
                plugin.register()
        self.assertIs(scheduler.restore_forward, mine)
        self.assertFalse(hasattr(scheduler, "hot_reload_forward"))


class TestHelpers(unittest.TestCase):
    def test_hot_reload_pins_the_file_and_returns_rank_zeros_answer(self):
        engine = FakeEngine()
        source = b"def make_forward(orig):\n    return orig\n"
        with tempfile.TemporaryDirectory() as d:
            candidate = Path(d) / "candidate.py"
            candidate.write_bytes(source)
            out = plugin.hot_reload_forward(engine, str(candidate))
        method, kwargs, before = engine.calls[0]
        self.assertEqual(method, "hot_reload_forward")
        self.assertEqual(
            kwargs,
            {
                "path": os.path.abspath(candidate),
                "sha256": hashlib.sha256(source).hexdigest(),
            },
        )
        self.assertEqual(before, [])  # a fresh directory per call
        self.assertEqual(out["rank"], 0)

    def test_restore_and_a_refusal(self):
        engine = FakeEngine()
        self.assertEqual(plugin.restore_forward(engine)["method"], "restore_forward")
        with self.assertRaisesRegex(AssertionError, "the engine is serving requests"):
            plugin.restore_forward(FakeEngine(fail="the engine is serving requests"))


if __name__ == "__main__":
    unittest.main()
