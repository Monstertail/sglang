"""Unit tests for plugin-registered scheduler RPCs.

Covers the registry itself and how ``Scheduler.handle_rpc_request`` dispatches:
registered handlers, built-in methods, return messages, errors and shadowing.

Run:  python -m pytest test/registered/unit/plugins/test_rpc_registry.py -v
"""

import types
from unittest.mock import patch

from sglang.srt.managers.io_struct import RpcReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.plugins.rpc_registry import get_rpc, register_rpc, unregister_rpc
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_NAMES = ("_test_rpc_echo", "_test_rpc_plain", "_test_rpc_boom", "flush_cache")


class _RpcTestCase(CustomTestCase):
    def setUp(self):
        for name in _NAMES:
            unregister_rpc(name)

    def tearDown(self):
        for name in _NAMES:
            unregister_rpc(name)


class TestRegistry(_RpcTestCase):
    def test_register_get_and_unregister(self):
        def echo(scheduler):
            return "x"

        self.assertIs(register_rpc("_test_rpc_echo", echo), echo)
        self.assertIs(get_rpc("_test_rpc_echo"), echo)
        unregister_rpc("_test_rpc_echo")
        self.assertIsNone(get_rpc("_test_rpc_echo"))

    def test_decorator_form(self):
        @register_rpc("_test_rpc_echo")
        def echo(scheduler):
            return "x"

        self.assertIs(get_rpc("_test_rpc_echo"), echo)

    def test_a_second_handler_for_a_name_is_refused(self):
        def first(scheduler):
            pass

        def second(scheduler):
            pass

        register_rpc("_test_rpc_echo", first)
        register_rpc("_test_rpc_echo", first)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_rpc("_test_rpc_echo", second)

    def test_a_handler_must_be_callable(self):
        with self.assertRaises(TypeError):
            register_rpc("_test_rpc_echo", "not callable")


class TestDispatch(_RpcTestCase):
    def call(self, scheduler, method, **parameters):
        req = RpcReqInput(method=method, parameters=parameters or None)
        with patch("sglang.srt.managers.scheduler.barrier"):
            return Scheduler.handle_rpc_request(scheduler, req)

    def scheduler(self, **attrs):
        return types.SimpleNamespace(
            tp_group=types.SimpleNamespace(cpu_group=None), **attrs
        )

    def test_a_registered_handler_gets_the_scheduler_and_its_string_is_the_message(
        self,
    ):
        seen = []

        @register_rpc("_test_rpc_echo")
        def echo(scheduler, *, value):
            seen.append((scheduler, value))
            return f"echo {value}"

        scheduler = self.scheduler()
        out = self.call(scheduler, "_test_rpc_echo", value=3)
        self.assertTrue(out.success)
        self.assertEqual(out.message, "echo 3")
        self.assertEqual(seen, [(scheduler, 3)])

    def test_a_result_that_is_not_a_string_gives_an_empty_message(self):
        register_rpc("_test_rpc_plain", lambda scheduler: {"not": "text"})
        out = self.call(self.scheduler(), "_test_rpc_plain")
        self.assertTrue(out.success)
        self.assertEqual(out.message, "")

    def test_a_raising_handler_reports_its_error(self):
        def boom(scheduler):
            raise RuntimeError("nope")

        register_rpc("_test_rpc_boom", boom)
        out = self.call(self.scheduler(), "_test_rpc_boom")
        self.assertFalse(out.success)
        self.assertEqual(out.message, "nope")

    def test_built_in_methods_still_dispatch(self):
        scheduler = self.scheduler(save_remote_model=lambda **kwargs: "saved")
        out = self.call(scheduler, "save_remote_model", url="x")
        self.assertTrue(out.success)
        self.assertEqual(out.message, "saved")

    def test_a_handler_that_shadows_a_scheduler_attribute_is_refused(self):
        register_rpc("flush_cache", lambda scheduler: "plugin")
        scheduler = self.scheduler(flush_cache=lambda: True)
        out = self.call(scheduler, "flush_cache")
        self.assertFalse(out.success)
        self.assertIn("shadows a Scheduler attribute", out.message)
