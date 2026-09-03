# Bare-metal GSM8K: LLaDA2.1-mini and Ling-mini-2.0

No Modal and no custom benchmark implementation: `gsm8k.sh` launches SGLang and calls the native `python3 -m sglang.test.run_eval` evaluator. This is the **unprofiled GSM8K benchmark**, not a Nsight trace capture.

## Clone and environment

```bash
git clone --single-branch --branch dllm-profiling-baremental \
  https://github.com/Monstertail/sglang.git sglang-baremental
cd sglang-baremental

# In a Linux GPU Python environment with a compatible NVIDIA driver:
python3 -m pip install -e ./python
```

The previous measurements used **one H100 80GB**. The source dependencies are specified by this checkout's `python/pyproject.toml`; installing from source also requires the build tools described in the repository installation docs (including Rust for the default native extensions). If you already have a compatible SGLang development environment, reuse it instead of reinstalling or upgrading the dependencies blindly. Do not substitute a random older SGLang release: it may lack this dLLM/FDFO path.

Use the same Python environment in both terminals. The script sets `PYTHONPATH` to **this clone's** `python/` automatically. It does not modify the installed package or launch paid cloud jobs. `SGLANG_PYTHON=/absolute/path/to/python` can select an environment explicitly.

The `run_eval --stop` change used by the original experiment is **already applied on this branch**. The adjacent `run_eval_stop.patch` is included for reference/porting to other checkouts; **do not apply it again here**.

## Run LLaDA

Terminal A, from the clone root:

```bash
bash benchmark/dllm/baremetal/gsm8k.sh serve-llada
```

Wait for model loading and CUDA Graph capture to finish. Then, in terminal B in the same clone and Python environment:

```bash
bash benchmark/dllm/baremetal/gsm8k.sh bench
```

The client first runs a short warmup, then the measured **50-question** evaluation. Read the **second** `Output throughput`, `Total latency`, and `Score` summary.

## Run Ling

Stop LLaDA with Ctrl-C in terminal A and wait for it to exit and release the GPU. Do not run both models on the same GPU at once.

```bash
# Terminal A
bash benchmark/dllm/baremetal/gsm8k.sh serve-ling

# Terminal B, after the server is ready
bash benchmark/dllm/baremetal/gsm8k.sh bench
```

## Settings

| Setting | LLaDA | Ling |
|---|---|---|
| Model | `inclusionAI/LLaDA2.1-mini` | `inclusionAI/Ling-mini-2.0` |
| Attention backend | `flashinfer` | `fa3` |
| CUDA Graph | on | on |
| FDFO | on, default `JointThreshold` | not applicable |
| CPU/GPU scheduling overlap | off | on |
| Radix cache | off | off |
| Client concurrency / server cap | 32 / 32 | 32 / 32 |
| GSM8K | 50 examples, 5-shot | identical questions |
| Prompt format | `--api chat`, official template | `--api chat`, official template |
| Generation | temperature 0, up to 512 tokens, natural EOS | same |
| Stop strings | `Question`, `Assistant:`, `<\|separator\|>` | same |

The models have their own chat templates/tokenizers, so equal questions do not imply equal input/output token counts. The concurrency cap does not imply an actual batch size of 32 throughout filling and draining.

To change concurrency, change it in **both** terminals:

```bash
# Terminal A
CONCURRENCY=1 bash benchmark/dllm/baremetal/gsm8k.sh serve-llada
# Terminal B
CONCURRENCY=1 bash benchmark/dllm/baremetal/gsm8k.sh bench
```

Optional environment variables: `CUDA_VISIBLE_DEVICES` (default `0`), `PORT` (default `30000`, same in both terminals), and `GSM8K_DATA_PATH` (an existing test JSONL for the client). Without a data path the native evaluator downloads GSM8K. Original dataset SHA256: `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.

Native reports:

```text
/tmp/gsm8k_inclusionAI_LLaDA2.1-mini.json
/tmp/gsm8k_inclusionAI_LLaDA2.1-mini.html
/tmp/gsm8k_inclusionAI_Ling-mini-2.0.json
/tmp/gsm8k_inclusionAI_Ling-mini-2.0.html
```

The native evaluator overwrites the same model's reports on a rerun; copy them elsewhere if you need to retain each result. `Output throughput` is output tokens/second, not prompt+output tokens/second. This non-streaming evaluator does not supply true streaming TTFT/TPOT. The entrypoint does not export custom per-request TPF diagnostics.

## Attention and reproducibility notes

The commands explicitly select FlashInfer for LLaDA and FA3 for Ling. Existing traces from other workloads showed `flashinfer::PrefillWithKVCacheKernel` / `flashinfer::BatchPrefillWithPagedKVCacheKernel` and `flash::FlashAttnFwdSm90`, respectively. The original GSM8K runs themselves were untraced. A Triton **MoE** warning does not mean Triton **attention**: MoE/sampling/auxiliary Triton kernels can still appear.

This minimal branch starts from `ed82bea1464d8ef66ed2b3ff6d9fc06c2e18ee60`, the same upstream base used in the original local profiling checkout. It adds the bare-metal entrypoint and evaluator stop-string routing, without the local Modal harness or optional TPF/NVTX instrumentation. No inference/scheduling/attention kernel logic is changed here. Hardware, package versions and numerical variation can change the results; this is not a promise of identical throughput.

Validated locally with shell syntax checks and CPU-only command/stop-routing tests, not a new GPU run:

```bash
bash -n benchmark/dllm/baremetal/gsm8k.sh
python3 benchmark/dllm/baremetal/test_entrypoint.py
python3 benchmark/dllm/test_eval_stop.py
```
