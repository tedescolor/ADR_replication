"""results/usage.csv: per kept run, wall minutes, turns, tool calls, input and output tokens, attempts, cost.

  python scripts/make_usage.py

Minutes, turns, tokens, attempts and cost come from results/per_fold.csv (written by score.py from each
run_summary.json); tool calls are counted in the archived transcript (agent_stream.jsonl.gz).
"""
import gzip
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, result_dir  # noqa: E402


def tool_calls(stream):
    n = 0
    with gzip.open(stream, "rt", errors="replace") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and rec.get("type") == "assistant":
                n += sum(1 for c in rec["message"].get("content", []) if c.get("type") == "tool_use")
    return n


def main():
    pf = pd.read_csv(RESULTS / "per_fold.csv")
    rows = []
    for r in pf.itertuples():
        d = result_dir(r.config, r.dataset, r.fold)
        task = json.loads((d / "task.json").read_text())
        rows.append({"config": r.config, "dataset": r.dataset, "fold": r.fold, "task": task["task"],
                     "attempt": task["attempt"], "minutes": r.minutes, "turns": r.turns,
                     "tool_calls": tool_calls(d / "agent_stream.jsonl.gz"),
                     "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                     "harness_attempts": r.attempts, "cost_usd": r.cost_usd})
    out = RESULTS / "usage.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"usage of {len(rows)} runs -> {out.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
