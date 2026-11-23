# Forward hot reload as an SGLang plugin

A running engine swaps its model forward for a candidate's, recaptures the
decode CUDA graphs and swaps back, without a restart. One warm engine can then
measure kernel candidates end to end instead of paying one cold start per
candidate.

This package does it through SGLang's general plugin system and changes no
SGLang file. The same feature built into SGLang is on branch
[`forward-hot-reload`](https://github.com/Monstertail/sglang/tree/forward-hot-reload),
with the same checks.

## Use

```sh
pip install -e examples/plugins/forward_hot_reload
export SGLANG_FORWARD_HOT_RELOAD=1   # without it the entry point does nothing
```

```python
import sglang as sgl
from sglang_forward_hot_reload import hot_reload_forward, restore_forward

engine = sgl.Engine(model_path=..., disable_prefill_cuda_graph=True)
hot_reload_forward(engine, "candidate.py")  # {"active": <sha256>, "capture_s": ...}
restore_forward(engine)                     # {"active": "original", "capture_s": ..., "env_moved": [...]}
```

```python
# candidate.py
import functools

def make_forward(original):
    @functools.wraps(original)  # SGLang inspects the forward's signature
    def forward(*args, **kwargs):
        return original(*args, **kwargs)
    return forward
```

## How it hooks in

| Needs | Done with |
|---|---|
| Run in every scheduler process | the `sglang.srt.plugins` entry point (`SGLANG_PLUGINS` can still filter it) |
| Two scheduler RPCs | methods added to `Scheduler`, which `Engine.collective_rpc` calls by name; an existing attribute of the same name is never replaced |
| A result back to the caller | `collective_rpc` returns nothing here, so every TP rank writes JSON into a directory the helper creates, and the helper returns rank 0's |
| Recapture | `ModelRunner.recapture_decode_cuda_graph()` |

## Checks

- **All ranks agree first.** Every TP rank passes the same preflight before any rank changes state:
  - the switch is on;
  - TP only, no DP-attention, no PP;
  - eager prefill;
  - the engine is idle;
  - the file's sha256 matches.
- **A failed capture rolls back.** If recapture fails on any rank, every rank goes back to the previous forward.
- **Unrecoverable cases stop the engine** (SystemExit):
  - a `make_forward` that moved or reshaped a parameter;
  - the original forward serving again (restored, or kept after a refused candidate or a rejected capture) while a torch precision or determinism flag, the default dtype, the current stream or `sys.path` differs from before the first swap.

  Environment variables that moved are reported by name only.

## Limits

- **Prefill CUDA graphs skip the swapped forward.** They capture the decoder layers, not `model.forward`, so the engine must run with `--disable-prefill-cuda-graph`, and a runner with a prefill graph is refused.
- **Flush the KV cache before comparing tokens.** Prefix-cache hits change the prefill path.
- **A guard failure stops the engine.** Run it in a child process.
- **Engine API only.** There is no HTTP endpoint.
- **Not covered:** speculative decoding (the draft model's forward is not swapped).
- **Version coupling.** The plugin relies on `Scheduler.handle_rpc_request` calling methods by name and on `ModelRunner.recapture_decode_cuda_graph`. A SGLang release that changes either breaks it, which is the price of not changing SGLang.

## Tests

```sh
python -m pytest examples/plugins/forward_hot_reload/tests -v
```

`test_forward_hot_reload.py` (two gloo ranks), `test_process_state_guard.py` and `test_plugin.py` need torch only. `test_e2e.py` needs SGLang and a GPU and skips otherwise.
