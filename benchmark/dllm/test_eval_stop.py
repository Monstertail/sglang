"""CPU-only tests of the native evaluator's explicit stop-string routing."""

import argparse
import ast
import json
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


SOURCE = Path(__file__).resolve().parents[2] / "python/sglang/test/run_eval.py"
TREE = ast.parse(SOURCE.read_text())


class Sampler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class StopRoutingTests(unittest.TestCase):
    def run_sampler(self, api, **extra):
        node = next(n for n in TREE.body if isinstance(n, ast.FunctionDef)
                    and n.name == "run_eval_once")
        namespace = dict(time=time, Eval=object, ChatCompletionSampler=Sampler,
                         CompletionSampler=Sampler, GenerateSampler=Sampler,
                         get_thinking_kwargs=lambda args: {}, parse_json_object=json.loads)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
        return namespace["run_eval_once"](
            SimpleNamespace(api=api, **extra), "http://127.0.0.1:30000/v1", lambda sampler: None
        )[2]

    def test_explicit_stops_all_apis(self):
        for api in ("chat", "completion", "generate"):
            with self.subTest(api=api):
                self.assertEqual(self.run_sampler(api, stop=["STOP"]).kwargs["stop"], ["STOP"])

    def test_omitted_chat_stop_unchanged(self):
        self.assertIsNone(self.run_sampler("chat").kwargs["stop"])

    def test_omitted_raw_stops_unchanged(self):
        for api in ("generate", "completion"):
            self.assertEqual(self.run_sampler(api).kwargs["stop"],
                             ["Question", "Assistant:", "<|separator|>"])

    def test_empty_stops_preserved(self):
        for api in ("chat", "completion", "generate"):
            self.assertEqual(self.run_sampler(api, stop=[]).kwargs["stop"], [])

    def test_cli_stop_parsing_and_default_suppression(self):
        call = next(n for n in ast.walk(TREE) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument"
                    and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--stop")
        parser = argparse.ArgumentParser()
        eval(compile(ast.Expression(call), str(SOURCE), "eval"),
             {"parser": parser, "argparse": argparse})
        self.assertFalse(hasattr(parser.parse_args([]), "stop"))
        self.assertEqual(parser.parse_args(["--stop", "Question", "Assistant:"]).stop,
                         ["Question", "Assistant:"])
        self.assertEqual(parser.parse_args(["--stop"]).stop, [])


if __name__ == "__main__":
    unittest.main()
