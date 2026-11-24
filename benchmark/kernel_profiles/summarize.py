"""Print a bench_modes.py run (its events.jsonl) as markdown: the banks, the checks,
the switches and every phase's metrics.

    python3 summarize.py OUT/events.jsonl
"""

import json
import sys
from pathlib import Path


def fmt(value, digits=3):
    return "" if value is None else round(value, digits)


def main(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    for r in rows:
        event = r.get("event")
        if event == "ready":
            resident = r["profiles"]["resident"]
            reason = f" ({resident['reason']})" if resident.get("reason") else ""
            print(f"ready in {r['ready_s']:.0f} s; resident graphs {'on' if resident['on'] else 'off'}{reason}")
            for name, p in r["profiles"]["profiles"].items():
                state = "available" if p["available"] else f"unavailable: {p['reason']}"
                print(f"- profile {name}: {state}; bank {p['bank']}")
        elif event in ("checks", "live_toggle"):
            fields = {k: v for k, v in r.items() if k not in ("at", "event", "summary")}
            print(f"{event}: {json.dumps(fields)}")
        elif event == "switch_failed":
            print(f"switch failed in {r['run']} {r['phase']} to {r['to']}: {r['error']}")
    switches = [r for r in rows if r.get("event") == "switch"]
    if switches:
        print("\n| run | phase | requests in flight | to | paused s | wall s |\n|---|---|---|---|---|---|")
        for r in switches:
            print(f"| {r['run']} | {r['phase']} | {r['load']} | {r['to']} | {fmt(r['paused_s'])} | {fmt(r['wall_s'])} |")
    print("\n| run | phase | s | requests | out tok/s | TTFT p50 s | TTFT p95 s | ITL mean ms | ITL p95 ms |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("event") == "summary":
            print(
                f"| {r['run']} | {r['phase']} | {r['seconds']} | {r['requests']} | {r['out_tok_s']} "
                f"| {fmt(r['ttft_p50'])} | {fmt(r['ttft_p95'])} | {fmt(r['itl_mean_ms'], 1)} | {fmt(r['itl_p95_ms'], 1)} |"
            )


if __name__ == "__main__":
    main(sys.argv[1])
