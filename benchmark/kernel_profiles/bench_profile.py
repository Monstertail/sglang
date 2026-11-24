"""Where a decode step's GPU time goes, at given batch sizes and prompt lengths (see README.md).

For each case (KP_CASES: JSON list of [name, users, prompt copies]) every user streams one request of
KP_OUT_TOKENS greedy tokens on `copies` shuffled copies of the prompt text. Once every request has
produced KP_START_AFTER tokens, the engine records KP_STEPS forward steps with the torch profiler (GPU
activity, CUDA graph replays included), and rank 0's trace is summed by kernel name into
OUT/kernels-<case>.json and by category into events.jsonl. Prefills are not deferred while profiling
(--prefill-decode-interval 0), so a case's users decode together.

Inputs as for bench_modes.py: KP_RECIPE, KP_PROMPT, KP_MODEL, KP_OUT, KP_SYS_PATH.
"""

import asyncio
import collections
import gzip
import json
import os
import random
import re
import sys
import time
from pathlib import Path

OUT = Path(os.environ.get("KP_OUT", "/out"))
RECIPE = Path(os.environ.get("KP_RECIPE", "/wf/recipe.json"))
PROMPT = Path(os.environ.get("KP_PROMPT", "/wf/prompt.txt"))
MODEL = os.environ.get("KP_MODEL", "/work/models/dsv41")
sys.path[:0] = [p for p in os.environ.get("KP_SYS_PATH", "/lean:/wf").split(":") if p]
CASES = json.loads(os.environ.get("KP_CASES", json.dumps(
    [["bs1-2k", 1, 1], ["bs16-2k", 16, 1], ["bs1-16k", 1, 8], ["bs16-16k", 16, 8]])))
STEPS = int(os.environ.get("KP_STEPS", "40"))
OUT_TOKENS = int(os.environ.get("KP_OUT_TOKENS", "2048"))
START_AFTER = int(os.environ.get("KP_START_AFTER", "64"))
# First match wins; the names are matched lower-cased.
CATEGORIES = [
    ("all-reduce", r"all_?reduce|allreduce|cross_device|rccl|nccl|quick_?reduce|reduce_scatter|all_?gather"),
    ("moe stage 1", r"stage1|moe_gemm1|gemm1"),
    ("moe stage 2", r"stage2|moe_gemm2|gemm2"),
    ("moe routing", r"moe|topk|expert|sorting|router|grouped_topk"),
    ("attention", r"attn|attention|flash|mla|sparse|indexer|paged|decode_kernel|mqa|swa|compress"),
    ("gemm", r"gemm|gemv|cijk|matmul|hipblaslt|tensile|ck_tile|mfma|_mm_|bf16_|fp8|fp4|a8w8|a4w4"),
    ("norm, rope, elementwise", r"norm|rope|rotary|elementwise|vectorized|copy|cat_|index|fill|reduce|act|silu|gelu"),
]


def log(event, **row):
    row = {"at": time.time(), "event": event, **row}
    with (OUT / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(row, default=str) + "\n")
    print(json.dumps(row, default=str)[:600], flush=True)


def chat_ids(tokenizer, text):
    try:
        import run_experiment

        return run_experiment.chat_input_ids(tokenizer, text)
    except ImportError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)


def category(name):
    lowered = name.lower()
    for label, pattern in CATEGORIES:
        if re.search(pattern, lowered):
            return label
    return "other"


def summed(trace_path):
    with gzip.open(trace_path, "rt") as f:
        events = json.load(f)["traceEvents"]
    kernels = collections.defaultdict(lambda: [0.0, 0])
    first, last = None, None
    for e in events:
        if e.get("ph") != "X" or e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        entry = kernels[e["name"]]
        entry[0] += float(e["dur"])
        entry[1] += 1
        first = e["ts"] if first is None else min(first, e["ts"])
        last = e["ts"] + e["dur"] if last is None else max(last, e["ts"] + e["dur"])
    return kernels, (0.0 if first is None else last - first)


async def stream(engine, ids, progress, index):
    out = await engine.async_generate(
        input_ids=ids, stream=True,
        sampling_params={"temperature": 0, "max_new_tokens": OUT_TOKENS, "ignore_eos": True})
    async for chunk in out:
        progress[index] = int(chunk["meta_info"].get("completion_tokens") or 0)


async def profile_case(engine, base, name, users, copies, tp):
    from sglang.srt.managers.io_struct import ProfileReq

    rng = random.Random(7)
    prompts = []
    for _ in range(users):
        ids = []
        for _ in range(copies):
            chunk = list(base)
            rng.shuffle(chunk)
            ids += chunk
        prompts.append(ids)
    progress = [0] * users
    tasks = [asyncio.create_task(stream(engine, p, progress, i)) for i, p in enumerate(prompts)]
    started = time.perf_counter()
    while min(progress) < START_AFTER and not all(t.done() for t in tasks):
        await asyncio.sleep(0.05)
    decoding_after = time.perf_counter() - started
    traces = OUT / "traces" / name
    traces.mkdir(parents=True, exist_ok=True)
    before = list(progress)
    await engine.tokenizer_manager.start_profile(ProfileReq(
        output_dir=str(traces), num_steps=STEPS, activities=["GPU"], with_stack=False,
        record_shapes=False, profile_prefix=name))
    files, sizes, waited = [], None, 0.0
    while waited < 600:
        files = sorted(traces.glob("*.trace.json.gz"))
        now = [f.stat().st_size for f in files]
        if len(files) >= tp and now == sizes:
            break
        sizes = now
        await asyncio.sleep(2.0)
        waited += 2.0
    tokens_during = [a - b for a, b in zip(progress, before)]
    await asyncio.gather(*tasks)
    rank0 = next((f for f in files if re.search(r"TP-?0\b|tp[-_]?0|rank[-_]?0", f.name)), files[0])
    kernels, wall_us = summed(rank0)
    busy = sum(v[0] for v in kernels.values())
    by_category = collections.defaultdict(float)
    for kernel, (dur, _) in kernels.items():
        by_category[category(kernel)] += dur
    top = sorted(kernels.items(), key=lambda kv: -kv[1][0])
    (OUT / f"kernels-{name}.json").write_text(json.dumps(
        {"trace": rank0.name, "steps": STEPS, "wall_us": wall_us, "busy_us": busy,
         "kernels": {k: {"us": v[0], "calls": v[1]} for k, v in top}}, indent=1))
    log("profile", case=name, users=users, prompt_tokens=len(prompts[0]), decoding_after_s=round(decoding_after, 1),
        tokens_during_profile=tokens_during, trace=rank0.name,
        step_wall_ms=round(wall_us / STEPS / 1000, 3), step_busy_ms=round(busy / STEPS / 1000, 3),
        share={k: round(v / busy, 4) for k, v in sorted(by_category.items(), key=lambda kv: -kv[1])},
        top=[[k[:120], round(v[0] / STEPS, 1), v[1] // STEPS, category(k)] for k, v in top[:40]])


def main():
    recipe = json.loads(RECIPE.read_text())
    os.environ.update(recipe["env"])
    argv = ["--enforce-shared-experts-fusion" if a == "--disable-shared-experts-fusion" else a
            for a in recipe["server_args"]]
    at = argv.index("--prefill-decode-interval")
    argv[at + 1] = "0"
    import sglang
    from sglang.srt.server_args import prepare_server_args

    started = time.monotonic()
    engine = sglang.Engine(server_args=prepare_server_args(["--model-path", MODEL, *argv]))
    log("ready", ready_s=time.monotonic() - started)
    tp = int(argv[argv.index("--tp") + 1])
    base = chat_ids(engine.tokenizer_manager.tokenizer, PROMPT.read_text())
    try:
        for name, users, copies in CASES:
            engine.loop.run_until_complete(profile_case(engine, base, name, users, copies, tp))
    finally:
        engine.shutdown()
    log("done")


if __name__ == "__main__":
    main()
