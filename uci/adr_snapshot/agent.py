"""Run an LLM agent that writes a conditional density program for your data.

    python -m adr.agent fit train.csv --target y --out runs/mydata
                        [--backend claude|ollama] [--model NAME] [--budget-min 60]
                        [--max-literals 80] [--keep-names]
    python -m adr.agent score runs/mydata/model.py test.csv --target y
    python -m adr.agent ollama-setup qwen3.8 [--num-ctx 65536]

`fit` creates a working directory with the training data (columns renamed x0.., y), a copy
of `adr` and TASK.md, runs one agent in it with headless Claude Code (`claude -p`), and checks
the program it writes against the contract. The agent chooses the exploration, validation
and model; the result is `<out>/model.py`, a pure function x -> adr.dist.Distribution:

    from adr.contract import load_model
    model = load_model("runs/mydata/model.py")
    d = model(x_row)          # d.logpdf(y), d.cdf(y), d.sample(rng, n)

Backends: `claude` uses your Claude Code login or ANTHROPIC_API_KEY; `ollama` points Claude
Code at a local Ollama server (Anthropic-compatible API), so the tools, prompt and task are
identical and only the model changes. Local models need a context window large enough for
Claude Code's system prompt; `ollama-setup` creates a variant with num_ctx=65536.

This module is also used by the UCI benchmark (experiments/uci/), whose TASK.md is rendered
from TASK_TEMPLATE below with the same text.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

PROMPT = ("You are a statistician solving a conditional density estimation task. Your working "
          "directory is {dir}. Read TASK.md there and follow it exactly: it describes the data, "
          "the output contract, the rules (work only inside that directory, OMP_NUM_THREADS=1, at "
          "most 2 processes) and the deliverables (model.py passing the contract check, and "
          "notes.md ending with an `expected_nll: <value>` line). Take the time you need to "
          "explore the data, compare candidate programs with honest out-of-sample validation, "
          "and refine. When done, reply with a short summary: the final program's structure, "
          "number of constants, your validation NLL and expected_nll.")

TASK_TEMPLATE = """# Task: conditional density estimation

`data.csv` has {n} rows: {d} covariates `x0`..`x{dm1}` and a target `y`.
Write `model.py` defining `model(x)` that, for one covariate row `x` (a numpy array of
length {d}, in the column order of `data.csv`), returns an `adr.dist.Distribution` for `y`.

{judged}
**Contract** (read `adr/dist.py` and `adr/contract.py`):
- `model.py` imports only from `adr.dist` and `math`; `model` is a pure function with its
  constants written in as numbers;
- at most {max_literals} constants and 500 AST nodes. Constants are counted as: every
  numeric or boolean literal and every use of `pi` or `e`; integer column indices such as
  the 3 in `x[3]` are not counted. String constants are not allowed (except docstrings);
- no loops, comprehensions, dicts or sets; no list/tuple literal longer than 6;
- check it with
  `python -m adr.contract model.py --features {d} --max-literals {max_literals} --data data.csv`
  (a model that fails the check scores nothing).

**Rules** (every action is logged and audited; a run that breaks them is discarded).
- Use only `data.csv` and your own analysis. No internet: do not use web search or fetch
  tools, do not download or install anything, do not call any network service.
- Work only inside this directory: put scratch files in `./work/`, not in /tmp or any
  scratch/temp directory, and do not read, list or search anything outside this directory.
- Do not try to identify the dataset or find other copies of it; treat it as anonymous.
- Use `{python}` (numpy, scipy, pandas, matplotlib). Run with `OMP_NUM_THREADS=1` and
  at most 2 processes at a time: other agents share the machine.

**Time budget: {budget} minutes** of wall-clock time from the start. Save a valid `model.py`
(and `notes.md`) early and overwrite them as you improve; at {budget} minutes the run is stopped
and whatever `model.py` is on disk is scored.

**Deliverables** (in this directory):
- `model.py`, passing the check;
- `notes.md`: what you tried, the validation scheme, why you chose the final program, and
  your estimate of its mean NLL on new data (one number, on its own line as
  `expected_nll: <value>`).
"""

DEFAULT_JUDGED = """**How you are judged.** The mean negative log-likelihood (natural log) of `model` on
held-out rows from the same distribution, which you do not have. Lower is better.
The density must be on the original scale of `y`: if you transform `y` while
exploring, the returned distribution must still be a density for the untransformed `y`.
Overfitting `data.csv` is your responsibility; how you validate is up to you.
"""

TOOLS = ["Bash", "Read", "Write", "Edit", "Glob", "Grep"]
DISALLOWED = ["WebFetch", "WebSearch", "Agent", "Task"]


DOMAIN_TXT = """
**Covariate domain.** `domain.csv` gives, for each covariate, the smallest and the largest value
in the full dataset, including the held-out rows on which `model` is scored. Every row on which
`model` is evaluated lies in this box. `model(x)` must return a valid distribution for every `x`
in the box, not only for the rows of `data.csv`: the contract check also evaluates it at points
of the box, including each covariate at its minimum and at its maximum.
"""


def prepare_task(workdir, X, y, max_literals=80, python=sys.executable, budget_min=60,
                 column_names=None, library=None, extra=None, judged=None, domain=None):
    """Write data.csv (x0.., y), a copy of the adr library and TASK.md into workdir.

    library: directory whose dist.py/contract.py are given to the agent (default: this package;
    the UCI benchmark passes its frozen snapshot, so that all its runs see the same library).
    extra: optional paragraph appended to TASK.md (e.g. that the target is discrete).
    judged: the scoring paragraph of TASK.md (default: the density NLL, DEFAULT_JUDGED).
    domain: optional (d, 2) array of [min, max] per covariate over the full dataset; written to
    domain.csv, described in TASK.md and enforced by the contract check (--domain)."""
    import pandas as pd
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    df = pd.DataFrame(X, columns=[f"x{j}" for j in range(X.shape[1])])
    df["y"] = y
    df.to_csv(workdir / "data.csv", index=False)
    shutil.copytree(library or HERE, workdir / "adr", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "agent.py"))
    task = TASK_TEMPLATE.format(n=len(y), d=X.shape[1], dm1=X.shape[1] - 1,
                                max_literals=max_literals, python=python, budget=budget_min,
                                judged=judged or DEFAULT_JUDGED)
    if domain is not None:
        domain = np.asarray(domain, dtype=float)
        pd.DataFrame({"column": [f"x{j}" for j in range(len(domain))], "min": domain[:, 0],
                      "max": domain[:, 1]}).to_csv(workdir / "domain.csv", index=False)
        task = task.replace("--data data.csv`", "--data data.csv --domain domain.csv`")
        task = task.replace("\n**Rules**", DOMAIN_TXT + "\n**Rules**", 1)
    if column_names is not None:
        task += "\n**Column meanings.** " + ", ".join(
            f"`x{j}` = {c}" for j, c in enumerate(column_names[:-1])) + \
            f", `y` = {column_names[-1]}.\n"
    if extra:
        task += "\n" + extra.strip() + "\n"
    (workdir / "TASK.md").write_text(task)
    return workdir


def backend_env(backend, base_url="http://localhost:11434", workdir=None):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1"}
    if backend == "ollama":
        env.update(ANTHROPIC_BASE_URL=base_url, ANTHROPIC_AUTH_TOKEN="ollama",
                   ANTHROPIC_API_KEY="")
        if workdir is not None:
            # Keep the harness's own session logs inside the task directory, so that they
            # are archived and locked with it (a local backend needs no login state).
            env["CLAUDE_CONFIG_DIR"] = str(Path(workdir) / ".claude_config")
    return env


RESUME = ("The previous turn ended with an error ({error}). Continue the task from where you "
          "stopped; the rules and deliverables in TASK.md are unchanged.")
MAX_ATTEMPTS = 20


def _results(path, start_line):
    """(session_id, last result record) among the stream records from start_line on."""
    sid, last = None, None
    with open(path) as f:
        for i, line in enumerate(f):
            if i < start_line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "system" and rec.get("subtype") == "init":
                sid = rec.get("session_id") or sid
            if rec.get("type") == "result":
                last = rec
    return sid, last


def agent_flags(model=None, settings=None):
    """Command-line flags of every agent session (shared by run_agent and the harness probe).

    Isolation: ``--tools`` restricts the session to TOOLS (``--allowedTools`` alone only
    pre-approves them); ``--restricted`` confines the file tools to the working directory and
    ignores user, project and local settings; ``--strict-mcp-config`` loads no MCP servers or
    connectors; ``--disable-slash-commands`` loads no skills. ``settings`` is a Claude Code
    settings file enabling the Bash sandbox (network denied, writes confined)."""
    flags = ["--tools", *TOOLS, "--allowedTools", *TOOLS, "--disallowedTools", *DISALLOWED,
             "--restricted", "--strict-mcp-config", "--disable-slash-commands",
             "--output-format", "stream-json", "--verbose"] + (["--model", model] if model else [])
    if settings:
        flags += ["--settings", str(Path(settings).resolve())]
    return flags


def run_agent(workdir, backend="claude", model=None, budget_min=60,
              base_url="http://localhost:11434", settings=None, tmpdir=None):
    """Run one headless agent in workdir within a wall-clock budget; returns a summary dict.

    If the run ends with an error of the harness or of the model server (e.g. an over-long
    response, or a transient server error) while budget remains, the same session is resumed
    with a neutral message (RESUME), up to MAX_ATTEMPTS times. Every attempt is appended to
    agent_stream.jsonl; the summary lists the errors and sums the usage over attempts.
    Flags: see agent_flags. ``tmpdir``: a private temporary directory for this session's harness
    files (CLAUDE_CODE_TMPDIR), so that the shared /tmp/claude-<uid> of other sessions is not
    visible inside its sandbox."""
    workdir = Path(workdir).resolve()
    flags = agent_flags(model, settings)
    env = backend_env(backend, base_url, workdir)
    if tmpdir:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
        env["CLAUDE_CODE_TMPDIR"] = env["TMPDIR"] = str(Path(tmpdir).resolve())
    stream = workdir / "agent_stream.jsonl"
    stream.write_text("")
    start, deadline = time.time(), time.time() + budget_min * 60
    cmd = ["claude", "-p", PROMPT.format(dir=workdir)] + flags
    status, errors, results, sid = "finished", [], [], None
    for attempt in range(MAX_ATTEMPTS):
        n0 = sum(1 for _ in open(stream))
        with open(stream, "a") as out, open(workdir / "agent_stderr.log", "a") as err:
            try:
                subprocess.run(cmd, cwd=workdir, stdout=out, stderr=err, env=env,
                               timeout=max(1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                status = "stopped at budget"
        new_sid, res = _results(stream, n0)
        sid = new_sid or sid
        if res:
            results.append(res)
        if status != "finished" or res is None or not res.get("is_error"):
            break
        errors.append(str(res.get("result", ""))[:200])
        if deadline - time.time() < 60 or not sid:
            status = "stopped after error"
            break
        cmd = ["claude", "-p", RESUME.format(error=errors[-1][:120]), "--resume", sid] + flags
    usage = {}
    for r in results:
        for k, v in (r.get("usage") or {}).items():
            if isinstance(v, (int, float)):
                usage[k] = usage.get(k, 0) + v
    summary = {"status": status, "minutes": round((time.time() - start) / 60, 1),
               "backend": backend, "model": model, "attempts": len(errors) + 1,
               "errors": errors, "num_turns": sum(r.get("num_turns") or 0 for r in results),
               "usage": usage, "cost_usd": sum(r.get("total_cost_usd") or 0 for r in results)}
    (workdir / "run_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def score(model_path, X, y):
    """Mean negative log-likelihood of a program on (X, y)."""
    from adr.contract import load_model
    model = load_model(model_path)
    lp = [float(np.asarray(model(x).logpdf(np.array([v])))[0]) for x, v in zip(X, y)]
    return -float(np.mean(lp))


def _read(path, target):
    import pandas as pd
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c != target]
    return df[cols].to_numpy(dtype=float), df[target].to_numpy(dtype=float), cols


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="run an agent on a training csv")
    f.add_argument("data")
    f.add_argument("--target", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--backend", choices=["claude", "ollama"], default="claude")
    f.add_argument("--model", default=None)
    f.add_argument("--budget-min", type=int, default=60)
    f.add_argument("--max-literals", type=int, default=80)
    f.add_argument("--keep-names", action="store_true",
                   help="tell the agent what each column means (default: anonymous)")
    f.add_argument("--base-url", default="http://localhost:11434")
    s = sub.add_parser("score", help="mean test NLL of a program")
    s.add_argument("model")
    s.add_argument("data")
    s.add_argument("--target", required=True)
    o = sub.add_parser("ollama-setup", help="create a large-context variant of an Ollama model")
    o.add_argument("model")
    o.add_argument("--num-ctx", type=int, default=65536)
    a = ap.parse_args()

    if a.cmd == "fit":
        from adr.contract import check
        X, y, cols = _read(a.data, a.target)
        out = prepare_task(a.out, X, y, a.max_literals, sys.executable, a.budget_min,
                           cols + [a.target] if a.keep_names else None)
        summary = run_agent(out, a.backend, a.model, a.budget_min, a.base_url)
        problems = check(out / "model.py", X.shape[1], 1, a.max_literals, xs=X) \
            if (out / "model.py").exists() else ["no model.py written"]
        summary["contract"] = "ok" if not problems else problems
        (out / "run_summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1))
    elif a.cmd == "score":
        X, y, _ = _read(a.data, a.target)
        print(f"mean NLL: {score(a.model, X, y):.4f}  (n={len(y)})")
    elif a.cmd == "ollama-setup":
        name = f"{a.model}-{a.num_ctx // 1024}k"
        modelfile = f"FROM {a.model}\nPARAMETER num_ctx {a.num_ctx}\n"
        subprocess.run(["ollama", "create", name, "-f", "-"], input=modelfile, text=True,
                       check=True)
        print(f"created {name}; use: python -m adr.agent fit ... --backend ollama --model {name}")


if __name__ == "__main__":
    main()
