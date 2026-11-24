"""Build a binary set: run the engine offline over the shapes it serves, with FlyDSL
and Triton compiling into an empty directory, then write its manifest (see
README.md and python/sglang/srt/managers/scheduler_components/kernel_binaries.py).

The --kernel-profiles file (KP_PROFILES) names the directory in "binaries". Its
FlyDSL cache starts as a copy of KP_BUILD_SEED (default: the cache aiter ships
precompiled), and Triton's as a copy of KP_BUILD_TRITON_SEED if set. Startup
compiles what model loading, graph capture and the profiles' banks still need.
Then, in every mode, single requests with each prompt length in KP_BUILD_LENGTHS
(1 to 16384 tokens, a factor of about 1.19 apart) and the batches in
KP_BUILD_BATCHES ([users, prompt length]), KP_BUILD_TOKENS output tokens each,
compile the eager shapes. The engine picks a kernel's launch configuration from the
token count, and the token count is not part of what a configuration compiles, so
a grid this dense meets every configuration a prefill can take. Every mode sends
prompts of its own and starts from an empty cache: a prompt the radix cache already
holds would prefill only its last tokens.

Inputs as for bench_modes.py: KP_RECIPE, KP_PROMPT, KP_MODEL, KP_PROFILES, KP_OUT,
KP_SYS_PATH.
"""

import asyncio
import json
import os
import random
import shutil
import sys
import time
import zlib
from pathlib import Path

OUT = Path(os.environ.get("KP_OUT", "/out"))
RECIPE = Path(os.environ.get("KP_RECIPE", "/wf/recipe.json"))
PROMPT = Path(os.environ.get("KP_PROMPT", "/wf/prompt.txt"))
MODEL = os.environ.get("KP_MODEL", "/work/models/dsv41")
PROFILES = os.environ.get("KP_PROFILES", "/kp/profiles.json")
sys.path[:0] = [p for p in os.environ.get("KP_SYS_PATH", "/lean:/wf").split(":") if p]
LENGTHS = json.loads(os.environ.get("KP_BUILD_LENGTHS", "null")) or sorted(
    {round(2 ** (i / 4)) for i in range(57)})
BATCHES = json.loads(os.environ.get(
    "KP_BUILD_BATCHES", "[[4, 512], [16, 512], [4, 2048], [16, 2048], [8, 4096]]"))
TOKENS = int(os.environ.get("KP_BUILD_TOKENS", "8"))


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


async def sweep(engine, base, mode):
    rng = random.Random(zlib.crc32(mode.encode()))
    for users, length in [(1, length) for length in LENGTHS] + [tuple(b) for b in BATCHES]:
        prompts = []
        for _ in range(users):
            ids = list(base) * (length // len(base) + 1)
            rng.shuffle(ids)
            prompts.append(ids[:length])
        await asyncio.gather(*(
            engine.async_generate(input_ids=p, sampling_params={
                "temperature": 0, "max_new_tokens": TOKENS, "ignore_eos": True})
            for p in prompts))


def main():
    from sglang.srt.managers.scheduler_components import kernel_binaries

    os.environ.pop(kernel_binaries.RUN_ONLY, None)
    spec = json.loads(Path(PROFILES).read_text())
    root = Path(spec["binaries"]["dir"])
    if kernel_binaries.entries(root, digest=False) or (root / kernel_binaries.MANIFEST).exists():
        raise SystemExit(f"{root} is not empty: build into an empty directory")
    seeds = {"flydsl": os.environ.get("KP_BUILD_SEED") or kernel_binaries.shipped_flydsl_cache(),
             "triton": os.environ.get("KP_BUILD_TRITON_SEED")}
    for cache, seed in seeds.items():
        if seed:
            shutil.copytree(seed, root / cache, dirs_exist_ok=True)
    log("seeded", seeds=seeds, entries=len(kernel_binaries.entries(root, digest=False)))
    recipe = json.loads(RECIPE.read_text())
    os.environ.update(recipe["env"])
    argv = ["--enforce-shared-experts-fusion" if a == "--disable-shared-experts-fusion" else a
            for a in recipe["server_args"]]
    argv += ["--kernel-profiles", PROFILES]
    import sglang
    from sglang.srt.server_args import prepare_server_args

    started = time.monotonic()
    engine = sglang.Engine(server_args=prepare_server_args(["--model-path", MODEL, *argv]))
    log("ready", ready_s=time.monotonic() - started, entries=len(kernel_binaries.entries(root, digest=False)),
        profiles=engine.get_kernel_profiles())
    base = chat_ids(engine.tokenizer_manager.tokenizer, PROMPT.read_text())
    try:
        for mode in sorted(spec.get("modes") or {}):
            began = time.monotonic()
            engine.switch_mode(mode)
            if getattr(engine.flush_cache(), "success", True) is False:
                raise RuntimeError("could not flush the cache before sweeping")
            engine.loop.run_until_complete(sweep(engine, base, mode))
            log("swept", mode=mode, seconds=round(time.monotonic() - began, 1),
                entries=len(kernel_binaries.entries(root, digest=False)))
    finally:
        engine.shutdown()
    manifest = kernel_binaries.write_manifest(
        root, server_args=argv, profiles=sorted(spec.get("profiles") or {}),
        modes=sorted(spec.get("modes") or {}), seeds=seeds, lengths=LENGTHS, batches=BATCHES, tokens=TOKENS)
    kernels = sorted({kernel_binaries.kernel(name) for name in manifest["entries"]})
    log("built", entries=len(manifest["entries"]), kernels=len(kernels),
        bytes=sum(e["bytes"] for e in manifest["entries"].values()), toolchain=manifest["toolchain"],
        flydsl=[k for k in kernels if k.startswith("flydsl:")][:40])


if __name__ == "__main__":
    main()
