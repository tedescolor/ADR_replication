"""Scan agent transcripts for signs that the agent identified its dataset or the benchmark.

Usage: python tools/contamination_scan.py <transcript files (.jsonl or .jsonl.gz)> [--out scan.csv]

Only text the agent itself produced is searched (its messages, its reasoning where recorded, and the
inputs of its tool calls, i.e. commands and files it wrote); the prompt, the task description and
tool outputs are not, so that a term injected by the harness does not count as use. Reports every
match of benchmark terms and of terms that would identify the source of the data (Wikipedia, page
views, edits, ...), with context, for manual review.
"""
import argparse
import gzip
import json
import re
from pathlib import Path

import pandas as pd

TERMS = {
    "benchmark": r"cpfn|push.?forward|franco|tedesco|\bkcde\b|conditional kde|arxiv|2511\.14455|\buci\b|cdtree",
    "source": r"wiki|mediawiki|wikimedia|pageviews?|page.?views?|\barticles?\b|edit.?counts?|number of edits|revisions?|"
              r"interlanguage|\bcategor(y|ies)\b|internal.?links?",
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
