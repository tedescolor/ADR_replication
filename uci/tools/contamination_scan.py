"""Scan agent transcripts for signs that the agent identified its dataset or the benchmark.

Usage: python experiments/uci/contamination_scan.py <transcript files (.jsonl or .jsonl.gz)> [--out scan.csv]

Only text the agent itself produced is searched (its messages, its reasoning where recorded, and the
inputs of its tool calls, i.e. commands and files it wrote); the prompt, the task description and
tool outputs are not, so that a term injected by the harness does not count as use. Reports every
match of benchmark terms (CDTree, UCI, the paper's authors) and of dataset-identifying terms, with
context, for manual review.
"""
import argparse
import gzip
import json
import re
from pathlib import Path

import pandas as pd

TERMS = {
    "benchmark": r"cdtree|histogram trees?|(?<![/\w])uci(?![/\w])|yang\s*(&|and)\s*van\s*leeuwen|van leeuwen",
    "dataset": (r"combined[\s_-]*cycle|power[\s_-]*plant|\bccpp\b|synchronous[\s_-]*machine|"
                r"qsar|aquatic|compressive[\s_-]*strength|slump[\s_-]*test|forest[\s_-]*fires?|"
                r"montesinho|naval|propulsion|gas[\s_-]*turbine|skillcraft|starcraft|"
                r"sml[\s_-]*2010|thermograph|support2|\bsupport[\s_-]*study|student[\s_-]*perf|"
                r"superconduct|critical[\s_-]*temp|indoor[\s_-]*localization|"
                r"clinicaltrials|land[\s_-]*registry|citi[\s_-]*bike|rideshare|\buber\b|\blyft\b"),
}


def records(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def agent_text(rec):
    """Text written by the agent in one transcript record."""
    if rec.get("type") != "assistant":
        return []
    out = []
    for c in rec.get("message", {}).get("content", []) or []:
        if c.get("type") in ("text", "thinking"):
            out.append(c.get("text") or c.get("thinking") or "")
        elif c.get("type") == "tool_use":
            out.append(json.dumps(c.get("input", {})))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for f in a.files:
        task = Path(f).name.split(".")[0]
        for rec in records(f):
            for text in agent_text(rec):
                for kind, pat in TERMS.items():
                    for m in re.finditer(pat, text, flags=re.I):
                        ctx = text[max(0, m.start() - 120): m.end() + 120].replace("\n", " ")
                        rows.append({"file": f, "task": task, "kind": kind,
                                     "term": m.group(0), "context": ctx})
    df = pd.DataFrame(rows, columns=["file", "task", "kind", "term", "context"])
    if a.out:
        df.to_csv(a.out, index=False)
    print(f"{len(a.files)} transcripts, {df.task.nunique()} with matches, {len(df)} matches")
    if len(df):
        print(df.groupby(["kind", "term"]).task.nunique().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
