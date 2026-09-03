#!/usr/bin/env bash
# Run with the SAME SGLang Python environment in the server and client terminals.
# Usage: bash gsm8k.sh serve-llada | serve-ling | bench
# Optional: CONCURRENCY=1 bash gsm8k.sh ... (set it in BOTH terminals).
set -euo pipefail

# Always run the Python package from this clone, not another installed checkout.
SGLANG_BAREMETAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_CHECKOUT_ROOT="$(cd "$SGLANG_BAREMETAL_DIR/../../.." && pwd)"
if [[ ! -f "$SGLANG_CHECKOUT_ROOT/python/sglang/__init__.py" ]]; then
    echo 'Run this script from its cloned benchmark/dllm/baremetal location.' >&2
    exit 2
fi
export PYTHONPATH="$SGLANG_CHECKOUT_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

CONCURRENCY="${CONCURRENCY:-32}"
PORT="${PORT:-30000}"
SGLANG_PYTHON="${SGLANG_PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if ! [[ "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
    echo 'CONCURRENCY must be a positive integer' >&2
    exit 2
fi

server_args=(
    --host 127.0.0.1 --port "$PORT"
    --tp 1 --trust-remote-code
    --mem-fraction-static 0.80
    --max-running-requests "$CONCURRENCY"
    --log-level warning
    --disable-radix-cache --enable-metrics
)

case "${1:-}" in
    serve-llada)
        # CUDA Graph is enabled by default; FDFO uses default JointThreshold settings.
        exec "$SGLANG_PYTHON" -m sglang.launch_server \
            --model-path inclusionAI/LLaDA2.1-mini \
            --dllm-algorithm JointThreshold --dllm-fdfo \
            --attention-backend flashinfer --disable-overlap-schedule \
            "${server_args[@]}"
        ;;
    serve-ling)
        # AR: CUDA Graph and overlap scheduling are enabled by default.
        exec "$SGLANG_PYTHON" -m sglang.launch_server \
            --model-path inclusionAI/Ling-mini-2.0 \
            --attention-backend fa3 \
            "${server_args[@]}"
        ;;
    bench)
        # This branch already includes the original experiment's --stop patch.
        eval_help="$("$SGLANG_PYTHON" -m sglang.test.run_eval --help)"
        if [[ "$eval_help" != *"--stop"* ]]; then
            echo 'run_eval lacks --stop. Check that you cloned dllm-profiling-baremental and use the intended Python environment.' >&2
            exit 2
        fi
        eval_args=(
            --host 127.0.0.1 --port "$PORT"
            --eval-name gsm8k --api chat
            --num-threads "$CONCURRENCY" --num-shots 5 --temperature 0
            --stop 'Question' 'Assistant:' '<|separator|>'
        )
        # Optional: point both models to the same downloaded GSM8K test.jsonl.
        if [[ -n "${GSM8K_DATA_PATH:-}" ]]; then
            eval_args+=(--gsm8k-data-path "$GSM8K_DATA_PATH")
        fi
        warmup_count="$CONCURRENCY"
        (( warmup_count >= 2 )) || warmup_count=2
        (( warmup_count <= 50 )) || warmup_count=50
        echo 'Warmup (do not use its throughput as the result):'
        "$SGLANG_PYTHON" -m sglang.test.run_eval "${eval_args[@]}" \
            --num-examples "$warmup_count" --max-tokens 64
        echo 'Measured GSM8K run: 50 examples, up to 512 output tokens each:'
        "$SGLANG_PYTHON" -m sglang.test.run_eval "${eval_args[@]}" \
            --num-examples 50 --max-tokens 512
        ;;
    *)
        echo 'Usage: bash gsm8k.sh {serve-llada|serve-ling|bench}' >&2
        exit 2
        ;;
esac
