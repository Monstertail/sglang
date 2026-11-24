"""Kernel profiles for DeepSeek-V4.1-Flash's FlyDSL MoE stage 1, for a GPU test of --kernel-profiles.

tuned: the tree a warm-arm agent left (ab-w4): stage 1's weight-scale loads streamed like the weights, 1.22-1.35x in
./bench and correct on the record's bound. slow: the stock with the A copy synchronous. counted: the stock. raises:
refuses to run under graph capture, so its switch has to roll back. Every replacement counts its calls by token count
and by whether a graph capture was running, in KP_OUT/calls-<pid>.json.
"""
import collections
import functools
import importlib.util
import json
import os
from pathlib import Path

import torch

OUT = Path(os.environ.get("KP_OUT", "/out"))
TREE = os.environ.get("KP_TUNED_TREE", "/trees/ab-w4")
TASK = os.environ.get("KP_TASK", "/lean/record/task.py")
LAYER = {"E": 385, "topk": 7, "K": 5120, "N": 1152, "act": 1}
SHAPES = [{**LAYER, "M": 6, "dtype": "fp8-sorted"}, {**LAYER, "M": 36, "dtype": "bf16"}]
TOLERANCE = {"t": 2.0 ** -7, "q": 2.0}
calls = collections.Counter()
launches = {}
LAUNCH_KEYS = ("tile_m", "tile_n", "tile_k", "out_dtype", "use_async_copy", "waves_per_eu", "b_nt", "k_wave",
               "xcd_swizzle", "v2_output_layout", "gate_mode", "topk")


def _counted(name, fn):
    @functools.wraps(fn)
    def stage1(a, *args, **kwargs):
        mode = "capture" if torch.cuda.is_current_stream_capturing() else "eager"
        key = f"{name} M{int(a.shape[0])} {mode}"
        calls[key] += 1
        launches.setdefault(f"M{int(a.shape[0])}", {k: str(kwargs.get(k, "default")) for k in LAUNCH_KEYS})
        n = sum(calls.values())
        if n in (1, 10, 100) or n % 200 == 0 or mode == "capture":
            (OUT / f"calls-{os.getpid()}.json").write_text(json.dumps({"calls": dict(calls), "launches": launches}))
        return fn(a, *args, **kwargs)

    return stage1


def make_double(original):
    """The stock run twice per call, the second time into a scratch output: a known-slower binary."""

    @functools.wraps(original)
    def stage1(*args, **kwargs):
        result = original(*args, **kwargs)
        args2, kwargs2 = list(args), dict(kwargs)
        if kwargs2.get("out") is not None:
            kwargs2["out"] = torch.empty_like(kwargs2["out"])
        elif len(args2) > 5 and args2[5] is not None:
            args2[5] = torch.empty_like(args2[5])
        # Same shapes and dtypes as the first call; only the output buffer differs (or the entry allocates one).
        original(*args2, **kwargs2)
        return result

    return _counted("double", stage1)


def make_counted(original):
    return _counted("stock", original)


def make_slow(original):
    @functools.wraps(original)
    def stage1(*args, **kwargs):
        kwargs["use_async_copy"] = False
        return original(*args, **kwargs)

    return _counted("slow", stage1)


def make_tuned(original):
    import shadow

    package, _ = shadow.load(TREE)
    return _counted("tuned", shadow.submodule(package, "moe_kernels").flydsl_moe_stage1)


def make_raises(original):
    @functools.wraps(original)
    def stage1(*args, **kwargs):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("raises_in_capture refuses to be captured")
        return original(*args, **kwargs)

    return stage1


def _task():
    spec = importlib.util.spec_from_file_location("kp_task_stage1", TASK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(pairs):
    """Each replacement against the record's fp64 reference and bound, at a decode and a short prefill shape."""
    task = _task()
    device = torch.device("cuda")
    for target, (_, replacement) in pairs.items():
        for s in SHAPES:
            inputs = task.make_inputs(s, "signed_spread", 1, device)
            outputs = {}
            for name, (size, dtype) in task.output_specs(s).items():
                if name != "out_scale":
                    t = torch.empty(size, dtype=dtype, device=device)
                    t.fill_(float("nan")) if dtype == torch.bfloat16 else t.zero_()
                    outputs[name] = t
            _, kwargs = task.prepare(s, inputs, outputs)
            task.run((replacement, kwargs), inputs, outputs, s)
            torch.cuda.synchronize()
            ref = task.reference(inputs, s)["out"]
            got, _ = task.decode("out", outputs["out"], None, outputs, inputs, s)
            limit = task.bound("out", ref, TOLERANCE, inputs, s)
            owned = ~torch.isnan(ref)
            bad = int((owned & ~((got.to(torch.float64) - ref).abs() <= limit)).sum())
            if bad:
                raise AssertionError(f"{target} at M{s['M']} {s['dtype']}: {bad} elements outside the bound")
            del inputs, outputs, ref, got, limit
    torch.cuda.empty_cache()
