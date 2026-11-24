"""Unit tests for kernel profiles across two TP ranks.

Two gloo processes on CPU; the model runner and the scheduler are stand-ins, the
kernels are module attributes, and a "captured graph" holds the kernel its capture
called and the metadata tensor of its batch size, and replays them.

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
from sglang.srt.model_executor import graph_banks
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
        "def make_graph_wrong(orig):\n"
        "    return lambda x: orig(x) if isinstance(x, int) else orig(x) + 1\n"
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
            # Passes its check, computes something else in its graphs.
            "graph_wrong": {
                "bindings": [
                    {"target": "kp_targets:kernel", "factory": "kp_impls:make_graph_wrong"}
                ],
                "check": "kp_impls:check",
            },
        },
        "modes": {
            "latency": {"scheduler": {"chunked_prefill_size": 4096, "prefill_decode_interval": 192}},
            "latency_big_chunks": {"scheduler": {"chunked_prefill_size": 16384}},
            "throughput": {
                "kernels": "low_latency",
                "scheduler": {"chunked_prefill_size": 16384, "prefill_decode_interval": 0},
            },
            "broken": {"kernels": "wrong"},
        },
    }


def resident(spec, names):
    return {
        **spec,
        "resident_graphs": True,
        "profiles": {n: p for n, p in spec["profiles"].items() if n in names},
        "modes": {
            n: m
            for n, m in spec["modes"].items()
            if m.get("kernels", kp.ORIGINAL) in (kp.ORIGINAL, *names)
        },
    }


class Graph:
    def __init__(self, kernel, metadata, scale):
        self.kernel, self.metadata, self.scale = kernel, metadata, scale

    def __call__(self):
        value = float(self.kernel(self.metadata) * self.scale)
        return types.SimpleNamespace(next_token_logits=torch.tensor([[value, -value]]))


class GraphBackend:
    def __init__(self):
        self._graphs, self._outputs = {}, {}

    @contextlib.contextmanager
    def replay_session(self):
        yield

    def replay(self, key, batch):
        return self._graphs[key]()


class Attention:
    attn_backend_list = None

    def __init__(self):
        self.cuda_graph_metadata_of_bucket_and_bs = {"verify": {}}
        self.workspace = torch.zeros(2)
        self.forward_metadata = None


class GraphRunner:
    """One graph per batch size; each reads its batch size's metadata, which the
    first capture of that size stores."""

    def __init__(self, model_runner):
        self.model_runner = model_runner
        self.backend = GraphBackend()
        self.attn_backend = Attention()
        self.scale = torch.ones(1)
        self.capture()

    def capture(self):
        import kp_targets

        kernel = kp_targets.kernel
        if self.model_runner.on_capture is not None:
            self.model_runner.on_capture(self)
        for bs in (1, 2):
            metadata = self.attn_backend.cuda_graph_metadata_of_bucket_and_bs["verify"]
            metadata = metadata.setdefault(bs, torch.full((1,), float(bs)))
            kernel(metadata)
            self.backend._graphs[bs] = Graph(kernel, metadata, self.scale)
            self.backend._outputs[bs] = None


class Runner:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.model.weight.data.fill_(3)
        self.fail_once = False
        self.prefill_graph = False
        self.on_capture = None
        self.spec_algorithm = types.SimpleNamespace(is_speculative=lambda: False)
        self.graph_memory_usage = {"decode": 0.5, "target_verify": 0.0}
        self.recapture_cuda_graphs()

    @property
    def captured(self):
        """The kernel the batch size 1 graph replays."""
        return self.decode_cuda_graph_runner.backend._graphs[1].kernel

    def has_prefill_cuda_graph(self):
        return self.prefill_graph

    def recapture_cuda_graphs(self):
        if self.fail_once:
            self.fail_once = False
            raise ValueError("capture rejected")
        self.decode_cuda_graph_runner = GraphRunner(self)
        self.captures = getattr(self, "captures", 0) + 1


def engine():
    runner = Runner()
    # value: nothing at all queued or running; kv: some request holds KV.
    idle = {"value": True, "kv": False, "flushes": 0}

    def flush_cache():
        idle["flushes"] += 1
        return True

    scheduler = types.SimpleNamespace(
        tp_group=types.SimpleNamespace(cpu_group=dist.group.WORLD),
        tp_worker=types.SimpleNamespace(model_runner=runner),
        flush_cache=flush_cache,
        is_fully_idle=lambda ignore_waiting=False: (
            not idle["kv"] if ignore_waiting else idle["value"]
        ),
        pause_state=types.SimpleNamespace(paused=False),
        chunked_prefill_size=8192,
        prefill_decode_interval=0,
        prefill_decode_interval_min_bs=1,
        _prefill_decode_interval_remaining=0,
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
            stack.enter_context(
                patch.object(graph_banks, "model_capture_mode", contextlib.nullcontext)
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

            # Modes: kernels and scheduler settings together, the startup value for
            # a setting a mode leaves out; a settings-only switch recaptures nothing.
            modes = json.loads(kp.list_kernel_profiles(scheduler))["modes"]
            assert modes["throughput"]["available"] and not modes["broken"]["available"]
            out = json.loads(kp.switch_mode(scheduler, name="throughput"))
            assert out["kernels"]["active"] == "low_latency" and runner.captured is kp_impls.fast
            assert (scheduler.chunked_prefill_size, scheduler.prefill_decode_interval) == (16384, 0)
            captures = runner.captures
            scheduler._prefill_decode_interval_remaining = 150
            out = json.loads(kp.switch_mode(scheduler, name="latency"))
            assert out["kernels"]["active"] == kp.ORIGINAL and runner.captures == captures + 1
            assert (scheduler.chunked_prefill_size, scheduler.prefill_decode_interval) == (4096, 192)
            assert scheduler._prefill_decode_interval_remaining == 150
            out = json.loads(kp.switch_mode(scheduler, name="latency_big_chunks"))
            assert out["kernels"] is None and runner.captures == captures + 1
            assert (scheduler.chunked_prefill_size, scheduler.prefill_decode_interval) == (16384, 0)
            assert scheduler._prefill_decode_interval_remaining == 0
            assert json.loads(kp.list_kernel_profiles(scheduler))["active_mode"] == "latency_big_chunks"
            with case.assertRaisesRegex(ValueError, "'wrong' are unavailable"):
                kp.switch_mode(scheduler, name="broken")

            # Live: only with generation paused and no request holding KV, and the
            # retracted requests' cached prefixes survive (no flush).
            plan = json.loads(kp.plan_mode(scheduler, name="throughput"))
            assert plan == {
                "mode": "throughput",
                "kernels_change": True,
                "resident": False,
                "paused": False,
                "holds_kv": False,
            }
            idle["value"] = False  # requests are waiting
            with case.assertRaisesRegex(RuntimeError, "needs generation paused"):
                kp.switch_mode(scheduler, name="throughput", live=True)
            scheduler.pause_state.paused, idle["kv"] = True, True
            with case.assertRaisesRegex(RuntimeError, "still holds KV"):
                kp.switch_mode(scheduler, name="throughput", live=True)
            idle["kv"] = False
            flushes = idle["flushes"]
            out = json.loads(kp.switch_mode(scheduler, name="throughput", live=True))
            assert out["kernels"]["live"] and runner.captured is kp_impls.fast
            assert idle["flushes"] == flushes
            out = json.loads(kp.switch_mode(scheduler, name="latency", live=True))
            assert runner.captured is original_kernel and idle["flushes"] == flushes
            scheduler.pause_state.paused, idle["value"] = False, True

            # A mode that names no profile or setting is refused when the file loads.
            for bad, message in (
                ({"kernels": "nope"}, "names no profile 'nope'"),
                ({"scheduler": {"max_running_requests": 4}}, "no scheduler setting"),
            ):
                with case.assertRaisesRegex(ValueError, message):
                    kp.KernelProfiles({"modes": {"m": bad}}, ())

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

            # Resident graphs: admission captures every available profile's graphs;
            # a switch selects them, serving or not, and recaptures nothing.
            runner, scheduler, idle = engine()
            state = kp.admit(scheduler, str(root / "profiles_resident.json"))
            listed = json.loads(kp.list_kernel_profiles(scheduler))
            p = listed["profiles"]
            assert listed["resident"]["on"], listed
            assert sorted(listed["resident"]["banks"]) == sorted([kp.ORIGINAL, "low_latency"])
            assert p["low_latency"]["bank"]["graphs"] == 2, p
            assert "graph bank: RuntimeError: cannot run under capture" in p["raises"]["reason"], p
            assert "logits are not the original's" in p["graph_wrong"]["reason"], p
            assert runner.captures == 1 and kp_targets.kernel is original_kernel
            graphs = runner.decode_cuda_graph_runner
            original_metadata = graphs.attn_backend.cuda_graph_metadata_of_bucket_and_bs
            idle["value"] = False
            out = switch(scheduler, "low_latency")
            assert out["resident"] and out["capture_s"] is None and runner.captures == 1, out
            assert runner.captured is kp_impls.fast and kp_targets.kernel is kp_impls.fast
            assert graphs.attn_backend.cuda_graph_metadata_of_bucket_and_bs is not original_metadata
            assert float(graphs.backend.replay(2, None).next_token_logits[0, 0]) == 4.0
            back = switch(scheduler, kp.ORIGINAL)
            assert back["previous"] == "low_latency" and runner.captured is original_kernel
            assert graphs.attn_backend.cuda_graph_metadata_of_bucket_and_bs is original_metadata
            plan = json.loads(kp.plan_mode(scheduler, name="throughput"))
            assert plan["kernels_change"] and plan["resident"], plan
            out = json.loads(kp.switch_mode(scheduler, name="throughput"))
            assert out["kernels"]["resident"] and runner.captured is kp_impls.fast
            assert runner.captures == 1
            switch(scheduler, kp.ORIGINAL)
            idle["value"] = True

            # A bank whose capture replaces state the earlier graphs read, or changes
            # what they compute, drops the banks: the engine recaptures and switches
            # by recapturing.
            for on_capture, message in (
                (
                    lambda g: setattr(g.attn_backend, "workspace", torch.zeros(2)),
                    r"replaced captured state: \['attn\[0\]\.workspace'\]",
                ),
                (lambda g: g.scale.mul_(2), "changed the original graphs' logits"),
            ):
                runner, scheduler, idle = engine()
                runner.on_capture = on_capture
                state = kp.admit(scheduler, str(root / "profiles_resident.json"))
                assert state.banks is None and runner.captures == 2, state.banks_reason
                case.assertRegex(state.banks_reason, message)
                assert state.profiles["low_latency"].available
                assert state.profiles["low_latency"].bank is None
                runner.on_capture = None
                out = switch(scheduler, "low_latency")
                assert out["capture_s"] is not None and runner.captured is kp_impls.fast
                switch(scheduler, kp.ORIGINAL)

            # Kernels that move process state under capture keep the server from starting.
            try:
                runner, scheduler, idle = engine()
                with case.assertRaisesRegex(
                    RuntimeError, "profiles' graphs moved process state: .*tf32_matmul"
                ):
                    kp.admit(scheduler, str(root / "profiles_resident_tf32.json"))
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
            (Path(d) / "profiles_resident.json").write_text(
                json.dumps(resident(profiles(d), ["low_latency", "wrong", "graph_wrong", "raises"]))
            )
            (Path(d) / "profiles_resident_tf32.json").write_text(
                json.dumps(resident(profiles(d), ["sets_tf32"]))
            )
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
