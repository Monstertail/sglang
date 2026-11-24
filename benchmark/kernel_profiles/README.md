# Kernel profiles: switching kernels and modes in a running engine

Record of the `kernel-profiles` branch (status 2026-11-24): what it adds, how to set it
up, how the GPU experiments ran, and what they measured, so that the work can be
repeated on another cluster.

The idea (from the discussion with Xiao): keep a set of kernel binaries for typical
kinds of traffic (low latency, high throughput) and move a running server among them,
efficiently and safely. The branch makes that a server feature:

- a **kernel profile** is a named set of kernel bindings (the binaries for one kind of
  traffic), admitted at startup: imported, compiled, checked;
- a **mode** is a profile plus the scheduler settings that go with it, the unit a
  workload-aware controller switches;
- a **graph bank** is a profile's decode graphs, captured at startup and kept, so a
  switch selects them instead of recapturing (`"resident_graphs": true`).

## Commits

| Commit | Adds |
|---|---|
| [b797d98](https://github.com/Monstertail/sglang/commit/b797d98) | Forward hot reload (`--enable-forward-hot-reload`): swap `model.forward` on every rank and recapture, for end-to-end kernel evaluation |
| [2e32855](https://github.com/Monstertail/sglang/commit/2e32855) | Forward hot reload checks process state when a refused candidate leaves the original serving |
| [3b5831a](https://github.com/Monstertail/sglang/commit/3b5831a) | Kernel profiles (`--kernel-profiles`): admission on every TP rank, switch by rebinding and recapturing, rollback |
| [4f4c466](https://github.com/Monstertail/sglang/commit/4f4c466) | Modes, and `Engine.async_switch_mode`: switch while serving (pause with retract, recapture, resume) |
| [b752352](https://github.com/Monstertail/sglang/commit/b752352a14f4bda7b0923989969ae8a412e0c317) | Resident graph banks: a switch selects captured graphs between two scheduler steps, with no pause |

Code: `python/sglang/srt/managers/scheduler_components/kernel_profiles.py`,
`python/sglang/srt/model_executor/graph_banks.py`, `Engine.switch_kernel_profile`,
`Engine.switch_mode`, `Engine.async_switch_mode`, `Engine.get_kernel_profiles`
(`python/sglang/srt/entrypoints/engine.py`). Unit tests:
`test/registered/unit/managers/test_kernel_profiles.py`.

## Using it

Launch with `--kernel-profiles profiles.json`. Requirements: tensor parallelism only (no
DP attention, no PP); eager prefill (`--cuda-graph-backend-prefill disabled` or
`--disable-prefill-cuda-graph`), since prefill graphs are not recaptured. For
DeepSeek-V4.1-Flash, `--enforce-shared-experts-fusion` makes decode call FlyDSL MoE
stage 1 (E385, top-7); with `--disable-shared-experts-fusion` decode takes another path
(E384, top-6) and a stage-1 profile changes nothing there.

```json
{
  "path": ["/kp"],
  "memory_margin_gb": 2.0,
  "resident_graphs": true,
  "bank_check": {"stable_rtol": 0.01, "stable_argmax": 0.02, "rtol": 0.05, "argmax": 0.1},
  "profiles": {
    "tuned": {
      "bindings": [{"target": "aiter.ops.flydsl.moe_kernels:flydsl_moe_stage1",
                    "factory": "kpkg.dsv41_stage1:make_tuned"}],
      "check": "kpkg.dsv41_stage1:check"
    }
  },
  "modes": {
    "latency":    {"kernels": "original", "scheduler": {"chunked_prefill_size": 4096,  "prefill_decode_interval": 192}},
    "throughput": {"kernels": "tuned",    "scheduler": {"chunked_prefill_size": 16384, "prefill_decode_interval": 0}}
  }
}
```

- `path` is prepended to `sys.path`; `bindings` replace entry points their callers look
  up at call time (`module:attr`), with `impl` (`module:attr`) or `factory`
  (`make(original) -> replacement`); `warmup(pairs)` and `check(pairs)` get
  `{target: (original, replacement)}`; both hooks are optional. `original` names every
  target's own entry point.
- Mode scheduler settings: `chunked_prefill_size`, `prefill_decode_interval`,
  `prefill_decode_interval_min_bs`; a setting a mode does not name returns to its
  startup value. They apply from the next batch.
- `bank_check` (optional) bounds how far a bank's logits may move on the capture inputs.

```python
engine.get_kernel_profiles()           # admission, banks, modes, recent switches
engine.switch_kernel_profile("tuned")  # idle engine, or any time with resident graphs
engine.switch_mode("throughput")
await engine.async_switch_mode("latency")  # while serving; pauses only without resident graphs
```

The RPCs behind them (`switch_kernel_profile`, `plan_mode`, `switch_mode`,
`list_kernel_profiles`) are registered with the plugin RPC registry.

### Writing a profile

A profile package is ordinary Python on `path`. Importing it must not change process
state (TF32, deterministic flags, ...) or rebind a target: either stops startup. A
replacement must compute what the original does; `check` should compare it against a
reference on the shapes the engine uses. `kpkg/dsv41_stage1.py` is the one the
experiments used:

- `make_tuned` loads an agent-optimized FlyDSL tree (`KP_TUNED_TREE`, mounted at
  `/trees/ab-w4`) under another package name with the warm-forward harness's
  `shadow.py` loader (not in this repository);
- `make_counted` is the stock kernel, wrapped to count calls by token count: a bank of
  it computes exactly what the original bank computes;
- `make_double` runs the stock kernel twice (a known-slower binary), `make_slow` turns
  off async copy, `make_raises` refuses graph capture (rollback tests);
- `check` compares against the stage-1 record's fp64 reference and error bound
  (`KP_TASK`, the record's `task.py`) at M=6 (fp8, sorted) and M=36 (bf16).

### Safety

- **Admission**, on every TP rank: a profile that fails to import, compile or check on
  any rank is unavailable, with the reason; the server serves `original`.
- **Switch by recapture** (no resident graphs): every rank must agree that the engine is
  idle (or paused with no request holding KV, for a live switch), that free memory
  covers the last capture plus the margin, and that no earlier switch left it broken. A
  failed capture rebinds the previous kernels and recaptures; if that fails, the engine
  stops. Returning to `original` checks process state; weight storage must not move.
- **Resident graphs**: each bank owns the graph backend's graphs and outputs, the
  attention backends' captured metadata (`cuda_graph_metadata_of_bucket_and_bs`), the
  metadata glue graphs and the ragged verify layouts, and shares the static input
  buffers, the logits buffer and the graph memory pool (one graph replays at a time, on
  one stream). At startup:
  - each bank keeps every tensor reachable from the runner and its backends, so no
    memory a graph reads is freed;
  - a capture that replaces one of those tensors (drift), or changes the original
    graphs' logits on the capture inputs, drops the banks: the engine recaptures its
    original graphs and switches by recapturing;
  - a profile whose graphs fail to capture, or whose logits differ from the original's
    beyond `bank_check`, is unavailable.

  A switch then rebinds the targets (for eager prefill) and swaps the bank between two
  scheduler steps. Banks are made for decode and DSpark verify runners with TP; other
  configurations fall back to switching by recapture, with the reason in
  `get_kernel_profiles()["resident"]`.

## Experiments

### Setup

- DeepSeek-V4.1-Flash, TP2 on two MI355X of one node, DSpark speculative decoding
  (block size 5), fused shared experts, eager prefill, target-verify graphs for batch
  sizes 1-16 (24 graphs with two attention variants), at most 16 running requests.
  Engine flags and environment: `recipe.dsv41-flash-tp2.json` (the driver turns
  `--disable-shared-experts-fusion` into `--enforce-shared-experts-fusion`).
- Image: sglang `de0f491836`, aiter `5405ee002`, FlyDSL 0.4.2 (digest
  `sha256:eae876433e6f...`). Its sglang is about a week older than this branch's base,
  so the branch was applied as a patch (see "Reproducing").
- Workload (`bench_modes.py`): four phases, low / high / low / high. Low: 2 users, a
  ~1.9k-token prompt, 40 s. High: 24 users, a ~3.8k-token prompt, 60 s. Each request:
  256 output tokens, greedy, `ignore_eos`, streamed; prompts are the shuffled tokens of
  one text, so DSpark accepts about 5 tokens per step.
- Modes: `latency` = chunked prefill 4096, prefill-decode interval 192, original
  kernels. `throughput` = chunked prefill 16384, interval 0, the tuned MoE stage 1
  (1.22-1.35x in the kernel benchmark). `throughput-settings` = the throughput
  settings with the original kernels.
- Controller: switch to the high mode when at least 8 requests are in flight for 2 s,
  back to `latency` when at most 3 are for 3 s, at most once per 5 s.
- Runs, in one engine after a warmup of both modes: always `latency`, always
  `throughput`, adaptive between `latency` and `throughput` (kernels change),
  adaptive between `latency` and `throughput-settings` (settings only).

### Results

Each cell gives the two phases of that load, first / second. ITL is the time per
output token (mean over requests, and the median of the requests' p95).

| Run | High: out tok/s | High: TTFT p95 (s) | Low: out tok/s | Low: TTFT p50 (s) | Low: ITL mean (ms) | Low: ITL p95 (ms) | Each switch |
|---|---|---|---|---|---|---|---|
| Always latency | 573 / 543 | 15.7 / 11.3 | 533 / 558 | 0.55 / 0.56 | 1.5 / 1.4 | 1.9 / 1.9 | |
| Always throughput (tuned kernels) | 2220 / 2228 | 1.61 / 1.66 | 871 / 860 | 0.10 / 0.10 | 1.9 / 1.9 | 4.2 / 4.2 | |
| Adaptive, kernels switch by recapture (4f4c466) | 1853 / 1529 | 10.9 / 10.1 | 544 / 454 | 0.55 / 0.56 | 1.5 / 2.5 | 2.3 / 2.5 | paused 7.0-7.5 s |
| Adaptive, kernels switch on resident graphs (b752352) | 2174 / 2174 | 1.84 / 1.76 | 550 / 593 | 0.55 / 0.54 | 1.5 / 1.5 | 2.0 / 2.0 | no pause, 19-63 ms |
| Adaptive, settings only | 2190 / 2215 | 1.78 / 2.10 | 552 / 588 | 0.55 / 0.54 | 1.5 / 1.5 | 2.0 / 2.0 | no pause, 13 ms |

The recapture row comes from an earlier run on two GPUs shared with other jobs. Its
adaptive phases fell outside the windows those jobs slowed; its always-throughput
phases matched these within 1-3%, its clean always-latency phases were about 10% lower.

- Neither fixed mode wins everywhere: under high load `throughput` gives 4x the output
  and a 7-10x lower TTFT p95; under low load `latency` keeps the per-token p95 at
  1.9 ms against 4.2 ms, at the cost of TTFT (0.55 vs 0.10 s), since it defers prefills.
- Adaptive with resident graphs serves within 2-3% of always-`throughput` under high
  load (the controller needs 2 s to see the load) and like always-`latency` under low
  load. Switching by recapture instead paused generation 7 s per switch and lost
  17-31% of high-load output.
- The tuned kernel adds nothing end to end (2174 vs 2190-2215 tok/s at the same
  settings): MoE stage 1 is about 12% of a verify step (running it twice cut verify
  steps from 129 to 114 per second), so a 1.3x stage 1 is worth about 3% at most. All
  the adaptive gain here comes from the scheduler settings.

### Where a decode step's time goes

`bench_profile.py`: steady decode with every request decoding (prefills not deferred),
40 steps recorded with the torch profiler on rank 0, CUDA graph replays included; a
step is the DSpark draft and the target verify. Original kernels. Share of GPU busy
time per step:

| Category | bs 1, 2k prompt | bs 16, 2k | bs 1, 16k | bs 16, 16k |
|---|---|---|---|---|
| MoE stage 1 (`mfma_moe1_*`) | 0.95 ms (12%) | 3.36 ms (22%) | 1.15 ms (14%) | 3.17 ms (21%) |
| MoE stage 2 (`gemm2_a4w4_*`) | 0.54 ms (7%) | 1.76 ms (11%) | 0.64 ms (8%) | 1.68 ms (11%) |
| MoE routing, sorting, quant | 0.67 ms (8%) | 1.12 ms (7%) | 0.65 ms (8%) | 1.11 ms (7%) |
| All-reduce (aiter Lamport; one-shot at bs 1, two-shot at bs 16) | 0.48 ms (6%) | 2.18 ms (14%) | 0.50 ms (6%) | 2.25 ms (15%) |
| Attention (sparse MLA, indexer) | 1.06 ms (13%) | 1.57 ms (10%) | 1.05 ms (13%) | 1.63 ms (11%) |
| Hyper-connections (`_hc_boundary_*`, Sinkhorn) | 1.33 ms (17%) | 1.45 ms (9%) | 1.31 ms (16%) | 1.46 ms (10%) |
| Dense GEMM + quant (fp8/mxfp8) | 2.23 ms (28%) | 3.05 ms (20%) | 2.19 ms (27%) | 3.03 ms (20%) |
| Norm, copy, elementwise | 0.51 ms (6%) | 0.63 ms (4%) | 0.50 ms (6%) | 0.62 ms (4%) |
| Other (sampling, rotary, page table, engram) | 0.21 ms (3%) | 0.22 ms (1%) | 0.25 ms (3%) | 0.27 ms (2%) |
| GPU busy / wall per step | 7.98 / 8.22 ms | 15.34 / 16.06 ms | 8.24 / 8.62 ms | 15.21 / 15.81 ms |

- The two regimes are bound by different kernels. At batch 1 (M=6, the latency
  mode's decode) dense GEMMs and hyper-connections take 45%; at batch 16 (M=96, the
  throughput mode's) MoE stage 1 and 2 and the all-reduce take 47%. A latency binary
  set and a throughput binary set would optimize different kernels.
- Stage 1's 12% at batch 1 matches the earlier estimate from running it twice; at
  batch 16 it is the largest single kernel (22%).
- Prompt length barely matters (2k vs 16k: within 5%): sparse attention reads a fixed
  number of blocks, so attention parameters chosen by context length are not a lever
  for this model at these lengths.
- 16x the batch costs 2x the step time: decode at batch 1 is far from compute-bound.

### Correctness of the banks

- Greedy outputs are not reproducible run to run in this configuration (the same
  request, alone, twice, can differ from the first token), so token-exact comparison
  across banks is not possible.
- Switching between the original bank and the stock-kernel bank (`same`, identical
  kernels) every 0.1 s during single greedy requests: DSpark's mean accepted length was
  4.29 tokens per step (12 log windows), against 3.77 without switching on the same
  prompts (37 windows) and 4.43 on the tuned bank (9). A bank reading metadata prepared
  for another would make verify disagree with the draft and the accepted length fall
  toward 1.
- No request returned fewer tokens than asked, with the mode flipping every second
  under eight users.
- The drift check fired once, on `candidate_masks` of the DSV4 HIP attention backend.
  That attribute hands candidates from one layer to later ones within a forward, and a
  graph makes and reads its own, so it is now excluded.

### What a bank costs

- Startup: one more capture per profile, 6.3 s for the stock-kernel bank and 9.0 s
  for the tuned one (24 graphs each, including the tuned kernel's compiles), plus a
  replay of each bank's graphs for the checks.
- Device memory: no change beyond ±0.1 GB (0.01 GB with graphs for batch sizes 1-2),
  because banks share the graph pool and the static buffers; a profile whose kernels
  need larger workspaces would grow the pool.
- A switch: one RPC to every rank and a reference swap, 13-63 ms of wall time on the
  caller, no pause on the GPUs.
- Banks are made at startup only; a new binary still takes a recapture or a restart.
  Draft graphs and prefill graphs are not banked.

### Observations

- First use of a kernel on a new shape compiles it while serving. With graphs only for
  batch sizes 1-2 (the smoke), eager verify of larger batches compiled the tuned kernel
  for each new M and a switch waited 32 s for the scheduler; with all graphs, the first
  prefills on the tuned kernel compiled and the first switch waited 17 s. Admission
  warmup should cover every M the engine will run (`warmup(pairs)`).
- Other jobs on the same GPUs slowed batch-size-1 decode from about 550 to 320 tok/s in
  some windows of the earlier run; `run_container.sh` refuses GPUs another container
  holds and stops its own container if one arrives.

## Reproducing on another cluster

Files here:

| File | What |
|---|---|
| `bench_modes.py` | The driver: checks, the loaded mode flips, and the four runs; writes `events.jsonl`, `results.json` |
| `bench_profile.py` | Decode-step kernel breakdown per batch size and prompt length; writes `kernels-<case>.json`, `events.jsonl` (needs no patch: original kernels) |
| `summarize.py` | Prints a run's `events.jsonl` as markdown tables |
| `run_container.sh` | Applies the branch to an image's sglang and runs the driver on two GPUs, with the guard |
| `profiles_resident.json` | Profiles `same`, `tuned` with resident graphs; modes `latency`, `same`, `throughput`, `throughput-settings` |
| `profiles_modes.json` | Profile `tuned` without resident graphs (switch by recapture) |
| `kpkg/dsv41_stage1.py` | The profile package |
| `recipe.dsv41-flash-tp2.json` | Engine flags and environment of the runs |
| `loader/sitecustomize.py` | Loads weights synchronously in every server process (HIP deadlock in the threaded loader on MI355X with ROCm 7.2) |

Not here, from the warm-forward harness: `shadow.py` (loads a second FlyDSL tree under
another package name, for `make_tuned`), the stage-1 record's `task.py` (for `check`),
the tuned FlyDSL tree, and the prompt text (any ~7 kB English text works). If the
harness directory also holds `run_experiment.py`, the driver uses its chat template;
otherwise the tokenizer's.

Steps:

1. Unit tests (CPU; from the repository root, with `PYTHONPATH=python` in an image
   that has sglang's dependencies):
   `python -m pytest test/registered/unit/managers/test_kernel_profiles.py test/registered/unit/managers/test_forward_hot_reload.py`
2. Pick two free GPUs; their render nodes are `/dev/dri/renderD128` + 8 per GPU on our
   nodes (GPU 6 = `renderD176`, GPU 7 = `renderD184`).
3. Smoke, graphs for batch sizes 1-2 only (smallest shapes first on a shared GPU):
   ```bash
   IMAGE=... WEIGHTS=... PROMPT=... HARNESS=... TREE=... OUT=/path/kp-smoke \
   GPUS="renderD176 renderD184" STAGE=smoke KP_MAX_BS=2 benchmark/kernel_profiles/run_container.sh
   python3 benchmark/kernel_profiles/summarize.py /path/kp-smoke/events.jsonl
   ```
   Expect resident graphs on, banks for `same` and `tuned`, `toggle_paused_s` 0.
4. Benchmark (about 30 min: 6 min startup, checks, 2 min warmup, four 4-minute runs):
   the same with `STAGE=bench`, no `KP_MAX_BS`, a new `OUT`.
5. Remove the container (`docker rm -f kp-bench`) when it is done; the guard exits.
6. Kernel breakdown: run `bench_profile.py` the same way (no patch needed; `KP_CASES`
   chooses batch sizes and prompt lengths), then read `kernels-<case>.json`.

Applying the branch to an older image (`run_container.sh` does this): `git diff 47361c5
HEAD -- python/` without `arg_groups/field_order.py`, `git apply` it in
`/sgl-workspace/sglang`, insert `"enable_forward_hot_reload"` and `"kernel_profiles"`
into `POSITIONAL_FIELD_ORDER`, and, if the scheduler has no
`scheduler_components/pause_state.py`, make `kernel_profiles._paused` read
`scheduler._engine_paused`. On an image built from this branch, `APPLY=0`.

## Next

- Binary sets that matter per regime, from the breakdown above: for throughput, MoE
  stage 1 and 2 and the all-reduce at M=84-96; for latency, the dense fp8 GEMMs and
  the hyper-connection kernels at M=6-12. Within one mode aiter already picks
  kernels per M (the two modes here ran M=6 and M=84-96 with different launch
  parameters), so a swap is needed where the sets cannot live in one table: other
  kernel families or fusions at a call site, global choices (speculative depth, CU
  partitioning for overlap), or a new version deployed without a restart.
- Speculative depth by load: upstream `--speculative-adaptive` picks the number of
  draft steps by batch size for EAGLE/EAGLE3, not DSpark.
- Admission warmup over every M; banks for the draft graphs; ahead-of-time kernel
  binaries instead of compiles at startup.
- The controller inside the scheduler, on queue depth and TTFT/ITL targets; an HTTP
  endpoint; upstream interfaces for exporting and installing a runner's graphs instead
  of `graph_banks` reaching into the runner.
