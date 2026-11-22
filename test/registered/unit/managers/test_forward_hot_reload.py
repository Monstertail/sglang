"""Unit tests for the forward hot-reload RPCs across two TP ranks.

Two gloo processes on CPU; the model runner and the scheduler are stand-ins and
the "captured graph" is the forward the runner held at its last recapture.

Run:  python -m pytest test/registered/unit/managers/test_forward_hot_reload.py -v
"""

import contextlib
import hashlib
import json
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.managers.scheduler_components import forward_hot_reload as fhr
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

CANDIDATES = {
    "plain": "def make_forward(orig):\n    return lambda x: orig(x) + 1\n",
    "tf32": (
        "import torch\n"
        "torch.backends.cuda.matmul.allow_tf32 = True\n"
        "def make_forward(orig):\n    return lambda x: orig(x) + 1\n"
    ),
    "weights": (
        "def make_forward(orig):\n"
        "    model = orig.__self__\n"
        "    model.weight.data = model.weight.data.clone()\n"
        "    return orig\n"
    ),
    "not_callable": "def make_forward(orig):\n    return 3\n",
    "tf32_not_callable": (
        "import torch\n"
        "torch.backends.cuda.matmul.allow_tf32 = True\n"
        "def make_forward(orig):\n    return 3\n"
    ),
}


class Runner:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.model.weight.data.fill_(3)
        self.fail_once = False
        self.prefill_graph = False
        self.recapture_cuda_graphs()

    def has_prefill_cuda_graph(self):
        return self.prefill_graph

    def recapture_cuda_graphs(self):
        if self.fail_once:
            self.fail_once = False
            raise ValueError("capture rejected")
        self.replay = self.model.forward


def engine():
    runner = Runner()
    scheduler = types.SimpleNamespace(
        tp_group=types.SimpleNamespace(cpu_group=dist.group.WORLD),
        tp_worker=types.SimpleNamespace(model_runner=runner),
        flush_cache=lambda: True,
        is_fully_idle=lambda: True,
    )
    return runner, scheduler


def rank_main(rank, root):
    root = Path(root)
    dist.init_process_group(
        "gloo",
        init_method=(root / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
    )
    case = CustomTestCase()
    x = torch.tensor([[2.0]])
    enabled = types.SimpleNamespace(enable_forward_hot_reload=True)
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("torch.cuda.synchronize"))
            stack.enter_context(
                patch.object(
                    fhr,
                    "get_exec",
                    return_value=types.SimpleNamespace(features=enabled),
                )
            )
            stack.enter_context(
                patch.object(
                    fhr,
                    "get_parallel",
                    return_value=types.SimpleNamespace(attn_dp_size=1, pp_size=1),
                )
            )

            def load(scheduler, name):
                path = root / f"{name}.py"
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                out = fhr.hot_reload_forward(scheduler, path=str(path), sha256=digest)
                return json.loads(out)

            # Out and back twice; nothing is left behind.
            runner, scheduler = engine()
            for _ in range(2):
                out = load(scheduler, "plain")
                assert out["active"] != fhr.ORIGINAL and out["capture_s"] >= 0, out
                assert runner.replay(x).item() == 7
                back = json.loads(fhr.restore_forward(scheduler))
                assert back["active"] == fhr.ORIGINAL and back["env_moved"] == []
                assert runner.replay(x).item() == 6
            assert runner.forward_hot_reload_slot.guard.moved()["flags"] == []

            # Refusals that leave the engine serving its forward.
            path = root / "plain.py"
            with case.assertRaisesRegex(RuntimeError, "does not match sha256"):
                fhr.hot_reload_forward(scheduler, path=str(path), sha256="0" * 64)
            with case.assertRaisesRegex(RuntimeError, "must return a callable"):
                load(scheduler, "not_callable")
            assert runner.replay(x).item() == 6
            assert not runner.forward_hot_reload_slot.broken

            # A capture rejected on one rank: every rank keeps the previous forward.
            runner.fail_once = rank == 1
            with case.assertRaisesRegex(RuntimeError, "kept the previous forward"):
                load(scheduler, "plain")
            assert runner.replay(x).item() == 6
            assert runner.forward_hot_reload_slot.active == fhr.ORIGINAL
            load(scheduler, "plain")
            assert runner.replay(x).item() == 7
            fhr.restore_forward(scheduler)

            # A candidate that turns TF32 on runs; the return to the original
            # forward stops the engine, and nothing more is accepted.
            try:
                runner, scheduler = engine()
                load(scheduler, "tf32")
                assert runner.replay(x).item() == 7
                with case.assertRaisesRegex(
                    SystemExit,
                    "original forward restored but .*tf32_matmul False -> True",
                ):
                    fhr.restore_forward(scheduler)
                assert runner.forward_hot_reload_slot.broken
                with case.assertRaisesRegex(RuntimeError, "earlier swap"):
                    load(scheduler, "plain")
            finally:
                torch.backends.cuda.matmul.allow_tf32 = False

            # The same candidate refused at build, or with its capture rejected
            # on one rank, has still run: the original forward keeps serving in
            # the state it left, so the engine stops.
            for name, rejecting_rank in (("tf32_not_callable", None), ("tf32", 1)):
                try:
                    runner, scheduler = engine()
                    runner.fail_once = rank == rejecting_rank
                    with case.assertRaisesRegex(
                        SystemExit,
                        "kept the original forward after a .* but "
                        ".*tf32_matmul False -> True",
                    ):
                        load(scheduler, name)
                    assert runner.forward_hot_reload_slot.broken
                    assert runner.forward_hot_reload_slot.active == fhr.ORIGINAL
                    assert runner.replay(x).item() == 6
                finally:
                    torch.backends.cuda.matmul.allow_tf32 = False

            # A factory that replaced a weight's storage stops the engine.
            runner, scheduler = engine()
            with case.assertRaisesRegex(SystemExit, "make_forward changed the weights"):
                load(scheduler, "weights")

            # A prefill CUDA graph would skip the swapped forward.
            runner, scheduler = engine()
            runner.prefill_graph = True
            with case.assertRaisesRegex(RuntimeError, "needs eager prefill"):
                load(scheduler, "plain")
            assert runner.replay(x).item() == 6

            # Without the server flag nothing is loaded.
            runner, scheduler = engine()
            enabled.enable_forward_hot_reload = False
            with case.assertRaisesRegex(RuntimeError, "--enable-forward-hot-reload"):
                load(scheduler, "plain")
            assert runner.replay(x).item() == 6
    finally:
        dist.destroy_process_group()


class TestForwardHotReload(CustomTestCase):
    def test_every_rank_agrees_and_rolls_back_together(self):
        with tempfile.TemporaryDirectory() as d:
            for name, text in CANDIDATES.items():
                (Path(d) / f"{name}.py").write_text(text)
            mp.spawn(rank_main, args=(d,), nprocs=2)
