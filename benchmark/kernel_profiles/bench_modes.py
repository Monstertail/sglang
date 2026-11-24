"""Microbenchmark of workload-aware mode switching (see README.md).

One engine runs the same phased workload four times: always in the latency mode, always in the throughput mode,
and twice with a controller that watches the requests in flight and switches modes while serving
(Engine.async_switch_mode): once to the throughput mode (other kernels), once to a mode with the throughput
scheduler settings and the original kernels. Every request streams; each phase reports TTFT, per-token latency
and output throughput, and each switch how long generation was paused. Before the runs, greedy checks and a
loaded run that flips modes every second (``KP_STAGE=smoke`` stops after them).

Inputs, by environment variable (defaults are the container mounts of run_container.sh):
  KP_RECIPE   JSON {"env": {...}, "server_args": [...]} of the engine to start   (/wf/recipe.json)
  KP_PROMPT   text the prompts are made of                                       (/wf/prompt.txt)
  KP_MODEL    model path                                                         (/work/models/dsv41)
  KP_PROFILES --kernel-profiles file                                             (/kp/profiles.json)
  KP_OUT      directory for events.jsonl and results.json                        (/out)
  KP_SYS_PATH extra import paths, ':'-separated                                 (/lean:/wf)
"""
import asyncio
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

OUT = Path(os.environ.get("KP_OUT", "/out"))
RECIPE = Path(os.environ.get("KP_RECIPE", "/wf/recipe.json"))
PROMPT = Path(os.environ.get("KP_PROMPT", "/wf/prompt.txt"))
MODEL = os.environ.get("KP_MODEL", "/work/models/dsv41")
PROFILES = os.environ.get("KP_PROFILES", "/kp/profiles.json")
sys.path[:0] = [p for p in os.environ.get("KP_SYS_PATH", "/lean:/wf").split(":") if p]
PHASES = json.loads(os.environ.get("KP_PHASES", json.dumps(
    [["low", 40, 2, 1], ["high", 60, 24, 2], ["low", 40, 2, 1], ["high", 60, 24, 2]])))  # name, s, users, prompt copies
WARMUP = [["high", 25, 24, 2], ["low", 15, 2, 1]]
OUT_TOKENS = int(os.environ.get("KP_OUT_TOKENS", "256"))
STAGE = os.environ.get("KP_STAGE", "bench")     # smoke: the correctness checks only
MAX_BS = os.environ.get("KP_MAX_BS")            # overrides --cuda-graph-max-bs-decode
CHECK_TOKENS = int(os.environ.get("KP_CHECK_TOKENS", "768"))
HIGH_AT, LOW_AT = 8, 3          # requests in flight
HIGH_FOR, LOW_FOR, DWELL = 2.0, 3.0, 5.0


def log(event, **row):
    row = {"at": time.time(), "event": event, **row}
    with (OUT / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(row, default=str) + "\n")
    print(json.dumps(row, default=str)[:400], flush=True)


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else None


class Run:
    def __init__(self, engine, base_ids, label, high_mode=None):
        self.engine, self.base, self.label, self.high_mode = engine, base_ids, label, high_mode
        self.inflight = 0
        self.records = []
        self.switches = []
        self.phase = None
        self.rng = random.Random(7)

    def prompt(self, copies):
        ids = []
        for _ in range(copies):
            chunk = list(self.base)
            self.rng.shuffle(chunk)
            ids += chunk
        return ids

    async def request(self, copies):
        ids = self.prompt(copies)
        phase = self.phase
        started = time.perf_counter()
        first = last = None
        tokens = 0
        gaps = []
        self.inflight += 1
        try:
            stream = await self.engine.async_generate(
                input_ids=ids, stream=True,
                sampling_params={"temperature": 0, "max_new_tokens": OUT_TOKENS, "ignore_eos": True})
            async for chunk in stream:
                now = time.perf_counter()
                n = int(chunk["meta_info"].get("completion_tokens") or 0)
                if n > tokens:
                    if first is None:
                        first = now
                    else:
                        gaps += [(now - last) / (n - tokens)] * (n - tokens)
                    last, tokens = now, n
        finally:
            self.inflight -= 1
        self.records.append({"phase": phase, "start": started, "end": last or time.perf_counter(),
                             "ttft": None if first is None else first - started, "tokens": tokens,
                             "itl_mean": statistics.mean(gaps) if gaps else None,
                             "itl_p95": pct(gaps, 0.95), "prompt": len(ids)})

    async def user(self, until, copies):
        while time.perf_counter() < until:
            await self.request(copies)

    async def controller(self, stop):
        mode, since_high, since_low, last_switch = "latency", None, None, 0.0
        while not stop.is_set():
            now = time.perf_counter()
            load = self.inflight
            since_high = (since_high or now) if load >= HIGH_AT else None
            since_low = (since_low or now) if load <= LOW_AT else None
            want = None
            if mode == "latency" and since_high and now - since_high >= HIGH_FOR:
                want = self.high_mode
            elif mode != "latency" and since_low and now - since_low >= LOW_FOR:
                want = "latency"
            if want and now - last_switch >= DWELL:
                began = time.perf_counter()
                try:
                    out = await self.engine.async_switch_mode(want)
                except Exception as error:
                    log("switch_failed", run=self.label, phase=self.phase, load=load, to=want, error=repr(error))
                    raise
                kernels = out.get("kernels") or {}
                row = {"run": self.label, "phase": self.phase, "load": load, "to": want,
                       "wall_s": time.perf_counter() - began, "paused_s": out.get("paused_s"),
                       "capture_s": kernels.get("capture_s")}
                self.switches.append(row)
                log("switch", **row)
                mode, last_switch = want, time.perf_counter()
            await asyncio.sleep(0.5)

    async def go(self, phases):
        stop = asyncio.Event()
        ctrl = asyncio.create_task(self.controller(stop)) if self.high_mode else None
        bounds = []
        for name, seconds, users, copies in phases:
            self.phase = f"{len(bounds)}:{name}"
            began = time.perf_counter()
            until = began + seconds
            await asyncio.gather(*(self.user(until, copies) for _ in range(users)))
            bounds.append((self.phase, began, time.perf_counter()))
            if ctrl and ctrl.done():
                ctrl.result()
        stop.set()
        if ctrl:
            await ctrl
        return self.summary(bounds)

    def summary(self, bounds):
        rows = []
        for phase, began, ended in bounds:
            recs = [r for r in self.records if r["phase"] == phase]
            done_tokens = sum(r["tokens"] for r in recs)
            rows.append({
                "run": self.label, "phase": phase, "seconds": round(ended - began, 1), "requests": len(recs),
                "out_tok_s": round(done_tokens / (ended - began), 1),
                "ttft_p50": pct([r["ttft"] for r in recs if r["ttft"] is not None], 0.5),
                "ttft_p95": pct([r["ttft"] for r in recs if r["ttft"] is not None], 0.95),
                "itl_mean_ms": 1000 * statistics.mean([r["itl_mean"] for r in recs if r["itl_mean"]]) if recs else None,
                "itl_p95_ms": 1000 * pct([r["itl_p95"] for r in recs if r["itl_p95"]], 0.5) if recs else None,
            })
        return rows


async def greedy(engine, ids, n):
    out = await engine.async_generate(
        input_ids=ids, sampling_params={"temperature": 0, "max_new_tokens": n, "ignore_eos": True})
    return list(out["output_ids"]) if out.get("output_ids") is not None else out["text"]


def first_diff(a, b):
    """Index of the first token where two greedy outputs differ; None if equal."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


async def toggling(engine, modes, every, stop, rows):
    i = 0
    while not stop.is_set():
        mode = modes[i % len(modes)]
        began = time.perf_counter()
        out = await engine.async_switch_mode(mode)
        rows.append({"to": mode, "wall_s": round(time.perf_counter() - began, 4), "paused_s": out["paused_s"],
                     "resident": (out.get("kernels") or {}).get("resident")})
        i += 1
        await asyncio.sleep(every)


async def checks(engine, prompts, label):
    """Greedy outputs: repeatable; the same when the stock kernel's graphs in another bank are
    switched in and out every 0.1 s mid-request; and after the tuned kernels served."""
    engine_mode = engine.async_switch_mode
    await engine_mode("latency")
    # The first run prefills in full and the rest from the radix cache, which rounds differently.
    cold = [await greedy(engine, p, CHECK_TOKENS) for p in prompts]
    ref = [await greedy(engine, p, CHECK_TOKENS) for p in prompts]
    again = [await greedy(engine, p, CHECK_TOKENS) for p in prompts]
    toggled, switches = [], []
    for p in prompts:
        stop, rows = asyncio.Event(), []
        task = asyncio.create_task(toggling(engine, ["same", "latency"], 0.1, stop, rows))
        toggled.append(await greedy(engine, p, CHECK_TOKENS))
        stop.set()
        await task
        switches.append(rows)
    await engine_mode("throughput")
    tuned = [await greedy(engine, p, CHECK_TOKENS) for p in prompts]
    await engine_mode("latency")
    after = [await greedy(engine, p, CHECK_TOKENS) for p in prompts]
    walls = sorted(r["wall_s"] for rows in switches for r in rows)
    row = {
        "label": label,
        "tokens": [len(r) for r in ref],
        "cold_first_diff": [first_diff(a, b) for a, b in zip(ref, cold)],
        "repeat_first_diff": [first_diff(a, b) for a, b in zip(ref, again)],
        "toggled_first_diff": [first_diff(a, b) for a, b in zip(ref, toggled)],
        "toggles": [len(rows) for rows in switches],
        "toggle_paused_s": max(r["paused_s"] for rows in switches for r in rows),
        "toggle_wall_ms_p50_max": [round(1000 * walls[len(walls) // 2], 2), round(1000 * walls[-1], 2)],
        "tuned_first_diff": [first_diff(a, b) for a, b in zip(ref, tuned)],
        "after_first_diff": [first_diff(a, b) for a, b in zip(ref, after)],
    }
    log("checks", **row)
    return ref


async def live_toggle(engine, base, prompts, ref):
    """Eight users for 20 s while the mode flips between latency and throughput every second."""
    run = Run(engine, base, "live-toggle")
    stop, rows = asyncio.Event(), []
    task = asyncio.create_task(toggling(engine, ["throughput", "latency"], 1.0, stop, rows))
    rows_summary = await run.go([["load", 20, 8, 1]])
    stop.set()
    await task
    await engine.async_switch_mode("latency")
    after = [await greedy(engine, p, CHECK_TOKENS) for p in prompts]
    log("live_toggle", switches=len(rows), paused_s=max(r["paused_s"] for r in rows),
        wall_ms_max=round(1000 * max(r["wall_s"] for r in rows), 2), summary=rows_summary,
        short=[r["tokens"] for r in run.records if r["tokens"] != OUT_TOKENS][:10],
        after_first_diff=[first_diff(a, b) for a, b in zip(ref, after)])


def chat_ids(tokenizer, text):
    try:
        # The warm-forward harness's template, which the measured runs used.
        import run_experiment

        return run_experiment.chat_input_ids(tokenizer, text)
    except ImportError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)


def main():
    recipe = json.loads(RECIPE.read_text())
    os.environ.update(recipe["env"])
    argv = ["--enforce-shared-experts-fusion" if a == "--disable-shared-experts-fusion" else a
            for a in recipe["server_args"]]
    if MAX_BS:
        at = argv.index("--cuda-graph-max-bs-decode")
        argv[at + 1] = MAX_BS
    argv += ["--kernel-profiles", PROFILES]
    import sglang
    from sglang.srt.server_args import prepare_server_args

    started = time.monotonic()
    engine = sglang.Engine(server_args=prepare_server_args(["--model-path", MODEL, *argv]))
    log("ready", ready_s=time.monotonic() - started, profiles=engine.get_kernel_profiles())
    tokenizer = engine.tokenizer_manager.tokenizer
    text = PROMPT.read_text()
    base = chat_ids(tokenizer, text)
    prompts = [chat_ids(tokenizer, text[:1500]), chat_ids(tokenizer, text[-3000:])]
    results = []
    try:
        ref = engine.loop.run_until_complete(checks(engine, prompts, "start"))
        engine.loop.run_until_complete(live_toggle(engine, base, prompts, ref))
        runs = () if STAGE == "smoke" else (("static-latency", "latency", None),
                                            ("static-throughput", "throughput", None),
                                            ("adaptive", "latency", "throughput"),
                                            ("adaptive-settings", "latency", "throughput-settings"))
        for mode in ("throughput", "latency") if runs else ():
            log("warmup", mode=mode, setup=engine.switch_mode(mode))
            engine.loop.run_until_complete(Run(engine, base, f"warmup-{mode}").go(WARMUP))
        for label, fixed, high_mode in runs:
            log("run", run=label, setup=engine.switch_mode(fixed))
            run = Run(engine, base, label, high_mode)
            rows = engine.loop.run_until_complete(run.go(PHASES))
            for row in rows:
                log("summary", **row)
            results.append({"run": label, "phases": rows, "switches": run.switches})
        log("final", profiles=engine.get_kernel_profiles())
    finally:
        (OUT / "results.json").write_text(json.dumps(results, indent=1, default=str))
        engine.shutdown()
    log("done")


if __name__ == "__main__":
    main()
