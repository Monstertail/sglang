"""CPU-only command tests; never import SGLang or launch a GPU server."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("gsm8k.sh")
ROOT = SCRIPT.resolve().parents[3]


class EntrypointTests(unittest.TestCase):
    def run_entrypoint(self, action, *, concurrency="32", help_has_stop=True):
        with tempfile.TemporaryDirectory() as directory:
            mock = Path(directory) / "python-mock"
            mock.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "if '--help' in sys.argv:\n"
                f"    print({'--stop' if help_has_stop else '--api'!r})\n"
                "else:\n"
                "    print(json.dumps({'args': sys.argv[1:], "
                "'pythonpath': os.environ['PYTHONPATH']}))\n"
            )
            mock.chmod(0o755)
            env = dict(os.environ, SGLANG_PYTHON=str(mock), CONCURRENCY=concurrency)
            env.pop("GSM8K_DATA_PATH", None)
            result = subprocess.run(
                ["bash", str(SCRIPT), action], cwd=directory,
                env=env, text=True, capture_output=True,
            )
            records = [json.loads(line) for line in result.stdout.splitlines()
                       if line.startswith("{")]
            return result, records

    def test_servers_use_explicit_attention_and_this_clone(self):
        for action, model, backend in (
            ("serve-llada", "inclusionAI/LLaDA2.1-mini", "flashinfer"),
            ("serve-ling", "inclusionAI/Ling-mini-2.0", "fa3"),
        ):
            with self.subTest(action=action):
                result, records = self.run_entrypoint(action)
                self.assertEqual(result.returncode, 0, result.stderr)
                args = records[0]["args"]
                self.assertEqual(args[args.index("--model-path") + 1], model)
                self.assertEqual(args[args.index("--attention-backend") + 1], backend)
                self.assertEqual(args[args.index("--max-running-requests") + 1], "32")
                self.assertIn("--disable-radix-cache", args)
                self.assertNotIn("--disable-cuda-graph", args)
                self.assertEqual(records[0]["pythonpath"].split(os.pathsep)[0],
                                 str(ROOT / "python"))
                self.assertEqual("--dllm-fdfo" in args, action == "serve-llada")
                self.assertEqual("--disable-overlap-schedule" in args,
                                 action == "serve-llada")

    def test_native_chat_warmup_then_50_questions(self):
        result, records = self.run_entrypoint("bench")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(records), 2)
        for record, count, max_tokens in zip(records, ("32", "50"), ("64", "512")):
            args = record["args"]
            self.assertEqual(args[:2], ["-m", "sglang.test.run_eval"])
            self.assertEqual(args[args.index("--api") + 1], "chat")
            self.assertEqual(args[args.index("--num-examples") + 1], count)
            self.assertEqual(args[args.index("--max-tokens") + 1], max_tokens)
            stop_index = args.index("--stop")
            self.assertEqual(args[stop_index + 1:stop_index + 4],
                             ["Question", "Assistant:", "<|separator|>"])

    def test_c1_has_two_warmup_questions(self):
        result, records = self.run_entrypoint("bench", concurrency="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = records[0]["args"]
        self.assertEqual(args[args.index("--num-examples") + 1], "2")
        self.assertEqual(args[args.index("--num-threads") + 1], "1")

    def test_missing_stop_support_fails_before_benchmark(self):
        result, records = self.run_entrypoint("bench", help_has_stop=False)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(records, [])

    def test_invalid_concurrency(self):
        result, records = self.run_entrypoint("serve-llada", concurrency="0")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
