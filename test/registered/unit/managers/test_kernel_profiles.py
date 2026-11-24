"""Unit tests for kernel profiles across two TP ranks.

Two gloo processes on CPU; the model runner and the scheduler are stand-ins, the
kernels are module attributes, and the "captured graph" is the kernel the runner
called and held at its last recapture.

Run:  python -m pytest test/registered/unit/managers/test_kernel_profiles.py -v
"""

import contextlib
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.managers.scheduler_components import kernel_profiles as kp
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

MODULES = {
    "kp_targets": "def kernel(x):\n    return x * 2\n\ndef other(x):\n    return x + 0\n",
    "kp_impls": (
        "import torch\n"
        "warmed = []\n"
        "def fast(x):\n    return x * 2\n"
        "def make_same(orig):\n    return lambda x: orig(x)\n"
        "def make_wrong(orig):\n    return lambda x: orig(x) + 1\n"
        "def make_tf32(orig):\n"
        "    def kernel(x):\n"
        "        torch.backends.cuda.matmul.allow_tf32 = True\n"
        "        return orig(x)\n"
        "    return kernel\n"
        "def make_raising(orig):\n"
        "    def kernel(x):\n"
        "        raise RuntimeError('cannot run under capture')\n"
        "    return kernel\n"
        "def warmup(pairs):\n    warmed.append(sorted(pairs))\n"
        "def warmup_fails(pairs):\n    raise RuntimeError('compile failed')\n"
        "def check(pairs):\n"
        "    for target, (orig, new) in pairs.items():\n"
        "        assert new(3) == orig(3), f'{target} differs from its original'\n"
    ),
    "kp_rebind": (
        "import kp_targets\n"
        "kp_targets.kernel = lambda x: x\n"
        "def fast(x):\n    return x * 2\n"
    ),
    "kp_tf32_import": (
        "import torch\n"
        "torch.backends.cuda.matmul.allow_tf32 = True\n"
        "def fast(x):\n    return x * 2\n"
    ),
}


def profiles(root):
    return {
        "path": [str(root)],
        "memory_margin_gb": 1.0,
        "profiles": {
            "low_latency": {
                "bindings": [
                    {"target": "kp_targets:kernel", "impl": "kp_impls:fast"},
                    {"target": "kp_targets:other", "factory": "kp_impls:make_same"},
                ],
                "warmup": "kp_impls:warmup",
                "check": "kp_impls:check",
            },
            "wrong": {
                "bindings": [{"target": "kp_targets:kernel", "factory": "kp_impls:make_wrong"}],
                "check": "kp_impls:check",
            },
            "failing_warmup": {
                "bindings": [{"target": "kp_targets:kernel", "impl": "kp_impls:fast"}],
                "warmup": "kp_impls:warmup_fails",
            },
            "rebinding": {
                "bindings": [{"target": "kp_targets:kernel", "impl": "kp_rebind:fast"}],
            },
            "sets_tf32": {
                "bindings": [{"target": "kp_targets:kernel", "factory": "kp_impls:make_tf32"}],
            },
            "raises": {
                "bindings": [{"target": "kp_targets:kernel", "factory": "kp_impls:make_raising"}],
            },
        },
    }


class Runner:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.model.weight.data.fill_(3)
        self.fail_once = False
        self.prefill_graph = False
        self.graph_memory_usage = {"decode": 0.5, "target_verify": 0.0}
        self.recapture_cuda_graphs()

    def has_prefill_cuda_graph(self):
        return self.prefill_graph

    def recapture_cuda_graphs(self):
        if self.fail_once:
            self.fail_once = False
            raise ValueError("capture rejected")
        import kp_targets

        # A capture runs the kernels once and replays what it ran.
        kp_targets.kernel(torch.tensor(1.0))
        self.captured = kp_targets.kernel


def engine():
    runner = Runner()
    idle = {"value": True}
    scheduler = types.SimpleNamespace(
        tp_group=types.SimpleNamespace(cpu_group=dist.group.WORLD),
        tp_worker=types.SimpleNamespace(model_runner=runner),
        flush_cache=lambda: True,
        is_fully_idle=lambda: idle["value"],
    )
    return runner, scheduler, idle


def rank_main(rank, root):
    root = Path(root)
    sys.path.insert(0, str(root))
    dist.init_process_group(
        "gloo", init_method=(root / "rendezvous").as_uri(), rank=rank, world_size=2
    )
    case = CustomTestCase()
    memory = {"free": 8 * 2**30}
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("torch.cuda.synchronize"))
            stack.enter_context(
                patch("torch.cuda.mem_get_info", side_effect=lambda: (memory["free"], 16 * 2**30))
            )
            stack.enter_context(
                patch.object(
                    kp,
                    "get_parallel",
                    return_value=types.SimpleNamespace(attn_dp_size=1, pp_size=1),
                )
            )
            import kp_impls
            import kp_targets

            original_kernel, original_other = kp_targets.kernel, kp_targets.other
            spec = root / "profiles.json"

            def switch(scheduler, name):
                return json.loads(kp.switch_kernel_profile(scheduler, name=name))

            # Admission: one profile ready, every broken one unavailable with why.
            runner, scheduler, idle = engine()
            state = kp.admit(scheduler, str(spec))
            listed = json.loads(kp.list_kernel_profiles(scheduler))
            p = listed["profiles"]
            assert listed["active"] == kp.ORIGINAL
            assert p["low_latency"]["available"] and p["sets_tf32"]["available"], p
            assert "differs from its original" in p["wrong"]["reason"], p
            assert "compile failed" in p["failing_warmup"]["reason"], p
            assert "rebound ['kp_targets:kernel']" in p["rebinding"]["reason"], p
            assert kp_targets.kernel is original_kernel
            assert kp_impls.warmed == [["kp_targets:kernel", "kp_targets:other"]]

            # Out and back twice; the graph replays what is bound.
            for _ in range(2):
                out = switch(scheduler, "low_latency")
                assert out["active"] == "low_latency" and out["previous"] == kp.ORIGINAL
                assert out["capture_s"] >= 0 and out["free_gb_before"] == 8.0, out
                assert runner.captured is kp_impls.fast
                assert kp_targets.other is not original_other
                assert switch(scheduler, "low_latency")["capture_s"] is None
                back = switch(scheduler, kp.ORIGINAL)
                assert back["previous"] == "low_latency"
                assert runner.captured is original_kernel
                assert kp_targets.other is original_other
            assert len(state.switches) == 6

            # Refusals that leave the original kernels serving.
            refusals = [
                ("nope", None, "no kernel profile 'nope'"),
                ("wrong", None, "is unavailable"),
                ("low_latency", "busy", "serving requests"),
                ("low_latency", "memory", "GB free; recapturing needs 1.50 GB"),
                ("low_latency", "prefill", "need eager prefill"),
            ]
            for name, condition, message in refusals:
                idle["value"] = condition != "busy"
                memory["free"] = 1 * 2**30 if condition == "memory" else 8 * 2**30
                runner.prefill_graph = condition == "prefill"
                with case.assertRaisesRegex(RuntimeError, message):
                    switch(scheduler, name)
                assert runner.captured is original_kernel and not state.broken
            idle["value"], memory["free"], runner.prefill_graph = True, 8 * 2**30, False

            # A capture rejected on one rank: every rank keeps the previous kernels.
            runner.fail_once = rank == 1
            with case.assertRaisesRegex(RuntimeError, "kept kernel profile 'original'"):
                switch(scheduler, "low_latency")
            assert runner.captured is original_kernel and state.active == kp.ORIGINAL
            switch(scheduler, "low_latency")
            runner.fail_once = rank == 1
            with case.assertRaisesRegex(RuntimeError, "kept kernel profile 'low_latency'"):
                switch(scheduler, kp.ORIGINAL)
            assert runner.captured is kp_impls.fast and state.active == "low_latency"
            switch(scheduler, kp.ORIGINAL)

            # A kernel that fails under capture is refused on every rank.
            with case.assertRaisesRegex(RuntimeError, "kept kernel profile 'original'"):
                switch(scheduler, "raises")
            assert runner.captured is original_kernel and not state.broken

            # Kernels that turn TF32 on run; coming back stops the engine.
            try:
                switch(scheduler, "sets_tf32")
                with case.assertRaisesRegex(
                    SystemExit, "original kernels restored but .*tf32_matmul False -> True"
                ):
                    switch(scheduler, kp.ORIGINAL)
                assert state.broken
                with case.assertRaisesRegex(RuntimeError, "earlier swap"):
                    switch(scheduler, "low_latency")
            finally:
                torch.backends.cuda.matmul.allow_tf32 = False

            # A profile whose import moves process state keeps the server from starting.
            try:
                runner, scheduler, idle = engine()
                with case.assertRaisesRegex(
                    RuntimeError, "moved process state: .*tf32_matmul False -> True"
                ):
                    kp.admit(scheduler, str(root / "profiles_tf32.json"))
            finally:
                torch.backends.cuda.matmul.allow_tf32 = False

            # Without --kernel-profiles there is nothing to switch.
            runner, scheduler, idle = engine()
            with case.assertRaisesRegex(RuntimeError, "--kernel-profiles"):
                switch(scheduler, "low_latency")
    finally:
        dist.destroy_process_group()


class TestKernelProfiles(CustomTestCase):
    def test_every_rank_admits_switches_and_rolls_back_together(self):
        with tempfile.TemporaryDirectory() as d:
            for name, text in MODULES.items():
                (Path(d) / f"{name}.py").write_text(text)
            (Path(d) / "profiles.json").write_text(json.dumps(profiles(d)))
            (Path(d) / "profiles_tf32.json").write_text(
                json.dumps(
                    {
                        "path": [d],
                        "profiles": {
                            "tf32": {
                                "bindings": [
                                    {"target": "kp_targets:kernel", "impl": "kp_tf32_import:fast"}
                                ]
                            }
                        },
                    }
                )
            )
            mp.spawn(rank_main, args=(d,), nprocs=2)
