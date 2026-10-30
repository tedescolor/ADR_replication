"""Audit the agents' transcripts for access outside their own task directory.

Usage: python experiments/uci/audit.py <run_dir> <transcript.jsonl> [...]
       python experiments/uci/audit.py <run_dir> --sandbox
       python experiments/uci/audit.py <run_dir> --streams

With --sandbox, the transcripts are <run_dir>/task*/agent_stream.jsonl written by
sandbox/run_sandboxed.sh, where each agent's own directory is /work (and /tmp is private).
With --streams, the same files written by run_agents.py, where each agent's own directory is
<run_dir>/<task>.

Each transcript is one subagent's JSONL log; its task is read from the first prompt. For
every task, reports web tool use, network-looking shell commands, and any path outside
<run_dir>/<task>/ that the agent read, wrote or mentioned in a command. Allowed: the venv
python, /dev/null, and paths the harness itself gave the agent in a tool result (e.g. the
log file of its own background job). Other agents' transcripts are never allowed. Writes <run_dir>/audit.csv and prints the non-clean tasks.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ALLOWED = [str(ROOT / ".venv/bin/python"), "/dev/null", "/usr/bin/env"]
WEB_TOOLS = {"WebFetch", "WebSearch"}
NETWORK = re.compile(r"\b(curl|wget|urllib|requests\.get|socket|pip install|uv pip|git clone|"
                     r"kaggle|ucimlrepo|openml|fetch_openml|sklearn\.datasets|https?://)")
ROOTS = "home|tmp|usr|etc|var|opt|root|mnt|media|srv|proc|sys|data|scratch|run|dev"
PATH = re.compile(rf"(?<![\w.)\]])(?:~/|/(?:{ROOTS}))(?:/[A-Za-z0-9_.~-]*)*|(?:\.\./){{2,}}\S*")


def records(transcript):
    """JSON records of a transcript; files that are not JSONL (e.g. shell logs) yield none."""
    for line in open(transcript, errors="replace"):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            return
        if isinstance(rec, dict):
            yield rec


def calls(transcript):
    for rec in records(transcript):
        if rec.get("type") != "assistant":
            continue
        for c in rec["message"].get("content") or []:
            if isinstance(c, dict) and c.get("type") == "tool_use":
                yield c["name"], c.get("input", {})


def harness_paths(transcript):
    """Paths that appear in the agent's own tool results (output of the harness)."""
    out = set()
    for rec in records(transcript):
        if rec.get("type") != "user" or not isinstance(rec["message"].get("content"), list):
            continue
        for c in rec["message"]["content"]:
            if isinstance(c, dict) and c.get("type") == "tool_result":
                for p in PATH.findall(json.dumps(c.get("content"))):
                    if "/tasks/b" in p:  # background-job logs only, never transcripts
                        out.add(p.rstrip("/.,;:)'\"\\"))
    return out


def task_of(transcript):
    for rec in records(transcript):
        if rec.get("type") == "user" and rec.get("parentUuid") is None:
            m = re.search(r"task\d+_f\d+", json.dumps(rec["message"]))
            return m.group(0) if m else None


def audit(run_dir, transcript, task=None, own=None):
    task = task or task_of(transcript)
    own = own or str(Path(run_dir).resolve() / task)
    given = harness_paths(transcript)
    row = {"task": task, "transcript": Path(transcript).name, "n_calls": 0,
           "web": [], "network": [], "outside": set()}
    for name, inp in calls(transcript):
        row["n_calls"] += 1
        if name in WEB_TOOLS or (name == "ToolSearch" and
                                 any(w.lower() in str(inp).lower() for w in WEB_TOOLS)):
            row["web"].append(f"{name}: {json.dumps(inp)[:120]}")
        text = inp.get("command", "") if name == "Bash" else \
            " ".join(str(inp.get(k, "")) for k in ("file_path", "path", "pattern", "notebook_path"))
        if name == "Bash" and NETWORK.search(text):
            row["network"].append(text[:160].replace("\n", " "))
        for p in PATH.findall(text):
            p = p.rstrip("/.,;:)'\"")
            if own == "/work" and p.startswith("/tmp"):
                continue
            if not p.startswith(own) and p not in given and \
                    not any(p.startswith(a) for a in ALLOWED):
                row["outside"].add(p)
    row["clean"] = not (row["web"] or row["network"] or row["outside"])
    row["outside"] = sorted(row["outside"])
    return row


def main():
    run_dir, transcripts = sys.argv[1], sys.argv[2:]
    if transcripts in (["--sandbox"], ["--streams"]):
        rows = [audit(run_dir, t, t.parent.name,
                      "/work" if transcripts == ["--sandbox"] else str(t.parent.resolve()))
                for t in sorted(Path(run_dir).glob("task*/agent_stream.jsonl"))]
    else:
        rows = [audit(run_dir, t) for t in transcripts if task_of(t)]
    df = pd.DataFrame(rows).sort_values("task")
    df.to_csv(Path(run_dir) / "audit.csv", index=False)
    print(f"{df.clean.sum()}/{len(df)} tasks clean")
    for r in df[~df.clean].itertuples():
        print(f"\n{r.task} ({r.n_calls} calls)")
        for k in ("web", "network", "outside"):
            for v in getattr(r, k):
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
