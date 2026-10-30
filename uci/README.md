# Agentic Distributional Regression: UCI benchmark

This file is the protocol of the study. It is also the set of instructions that the Claude Code
session orchestrating the runs received (`PROMPT.txt`): where it does not decide something, the
orchestrator asks the person running the study and never improvises a protocol choice.

**Goal.** Add agent columns to Table 3 of Yang & van Leeuwen, *Conditional Density Estimation
with Histogram Trees* (NeurIPS 2024, arXiv:2410.11449): the 12 of its 14 UCI datasets whose target is
continuous, their data, their five folds. The score is the **ordinary test negative
log-likelihood** (density NLL) on the released files, the same score as the published competitors
(§3, Score), stated to the agents in advance. One **agent configuration** is evaluated, with a
60-minute budget:

| config | model | model id | budget |
|---|---|---|---|
| `opus-60` | Claude Opus 5.5 | `claude-opus-5-5` | 60 min |

**Covariate domain.** As Yang & van Leeuwen set the range of their histograms from the full dataset
before the train/test split, every task directory contains `domain.csv`: the minimum and maximum of
each covariate over the full dataset (all folds). `TASK.md` states that every row on which the program
is evaluated lies in this box, and the contract check requires a valid distribution throughout it
(§1.2). Every test row can therefore be scored.

All methods are compared with the same score, the ordinary test NLL of the published columns.
Programs that exploit the noise the CDTree authors added to the data are rewarded like any other
well-fitting density: that is part of the benchmark as released.

Agent runs cannot be reproduced token by token. The replication package therefore has two
parts: (i) every program, note and transcript, collected and released; (ii) every table and
number regenerated **deterministically from the released programs** by one script.

### Contents of this folder (self-contained; nothing has to be downloaded)

| path | what it is |
|---|---|
| `adr/` | the ADR library: `dist.py` (distribution families), `contract.py` (output checker), `agent.py` (task template, `prepare_task`, headless `run_agent`) |
| `data/uci/` | the 14 datasets as released by the CDTree authors (`permuted_dataset_noiseadded`, commit `b607e12`), unchanged; `SHA256SUMS`, `SOURCE.txt` |
| `data/uci_original/` | the same rows before the authors added noise (`permuted_dataset`, same commit); used **only** by the scorer and `scripts/make_precision.py`, never given to an agent; `SHA256SUMS`, `SOURCE.txt` |
| `data/uci_precision.csv` | the recording precision `w` of every target, fixed before the runs (`scripts/make_precision.py`; checked by `setup.sh`) |
| `published/table3_nll.csv` | Table 3 of Yang & van Leeuwen (2024), transcribed, with `SOURCE.txt` |
| `scripts/` | **ready-to-run pipeline — use it, do not rewrite it**: `setup.sh` (step 1), `probe.py` (step 2), `wave.py` (prepare / run / audit / archive / rerun a fold), `score.py`, `make_tables.py`, `reproduce.sh`, `fix_apparmor.sh` (operator, sudo), `common.py` (paths, datasets, configs, folds) |
| `sandbox_settings.json` | template of the sandbox settings given to every agent (installed by `setup.sh`) |
| `tools/` | `audit.py` (transcript audit, used by `wave.py audit`), `contamination_scan.py` |
| `requirements.txt` | pinned Python environment |

The machine needs Python 3.11, and the `claude` CLI logged in with access to both models.
Internet is needed only to install the pinned Python packages.

---

## 1. The method and the estimator class (ADR)

An agent receives training data `(x_i, y_i)` and returns a Python program `model.py` that
defines `model(x)`. It maps one covariate vector `x` (a 1-d numpy array) to a **distribution
object** for `y`. The program is the estimate `p̂(y | x)`. It must be a proper conditional
probability distribution, with exact `logpdf`, `cdf`, `ppf` and `sample`.

### 1.1 Library `adr.dist` (the only building blocks)

Copy `adr/` unchanged to `adr_snapshot/`, and freeze it for the whole analysis. Every task
receives an identical copy, without `agent.py`. **Never change the library during the
analysis.**

- **Univariate families**, all with array-valued parameters:
  - `Normal(loc, scale)`
  - `StudentT(df, loc, scale)`
  - `SkewNormal(loc, scale, shape)`
  - `Laplace(loc, scale)`
  - `LogNormal(mu, sigma)`
  - `Gamma(shape, rate)`
- **Mixtures:** `Mixture(weights, components)` has at most 5 components and is nestable.
- **Multivariate constructors:** `MultivariateNormal`, `Independent` and `Sequential` exist in
  the library, but they are not used here, since every target is univariate.
- **Elementary functions:** `sin`, `cos`, `exp`, `log`, `sqrt`, `tanh`, `arctan2`, `abs`,
  `maximum`, `minimum`, `where`, `sigmoid`, `softmax`, `dot`, and the constant `pi`.
- **Scale floor:** every scale parameter is floored at `SIGMA_MIN = 1e-6`.

The class therefore contains:
- GAMLSS-type models, with parametric terms for every distribution parameter;
- gated mixtures of experts;
- trees with parametric leaves, written as `if`/`where`;
- small neural networks written out explicitly.

### 1.2 Output contract (`adr.contract`, checked by machine)

A program is accepted only if both parts pass.

- **Static part:**
  - Imports come only from `adr.dist` and `math`.
  - `model` is a pure function.
  - No loops, comprehensions, dicts, sets, I/O, `eval`, `exec` or private attributes.
  - List and tuple literals have at most 6 elements.
  - At most **500 AST nodes**.
  - At most **K = 80 constants**. Constants are every numeric or boolean literal and every
    use of `pi` or `e`. Bare integer subscripts (`x[3]`, `x[0:5]`, `x[[0, 7]]`) are free, but
    integers in index arithmetic are counted. String constants are rejected, except
    docstrings.
- **Dynamic part:** on 20 covariate rows of the training data, `model` returns a
  `Distribution` whose samples have finite log-density.
- **Covariate domain** (`--domain domain.csv`): the same holds at both corners of the covariate box, at
  50 uniform points in it and, starting from 5 of the training rows, with each covariate set in turn to
  its minimum and to its maximum.

Check a program with:

```
python -m adr.contract model.py --features d --max-literals 80 --data data.csv --domain domain.csv
```

The contract fixes **what** is returned, not **how** it is found. The whole statistical
workflow belongs to the agent: exploration, validation scheme, candidate structures,
estimation, final refit.

### 1.3 Idealized procedure (for the paper's theory; do not impose it on the agents)

1. Split the data into a fit part and a validation part.
2. Propose structures `s_1, …, s_M` adaptively.
3. Fit the constants of each by maximum likelihood.
4. Select by held-out log-loss.
5. Optionally refit on all the data.
6. Report the program and the agent's estimate of its test NLL.

---

## 2. Agent protocol

### 2.1 Task directory

One task is one (dataset, fold). The directory contains exactly these files:

- `data.csv`: the training rows of that fold only, with covariates renamed `x0 … x{d-1}` and
  the target renamed `y`.
- `adr/`: a copy of `adr_snapshot/`, excluding `agent.py`.
- `TASK.md`: rendered by `adr.agent.prepare_task` with the task's `n`, `d`, `K = 80`, the
  Python path and the **config's budget** (60).

**Test rows are never written to any task directory.** They are recomputed from `data/uci/`
only at scoring time.

**Use neutral names.** Everything an agent can see — its task path, shown in the prompt, and
the Python path, shown in TASK.md — must be outside `uci/`. Use one neutral workspace:

- **Python environment:** `~/ws/env/`, so the agents call `~/ws/env/bin/python`. Create it
  from `uci/requirements.txt`.
- **Live task directories:** `~/ws/runs/<config>/t0001`, `t0002`, …

Keep the mapping from task number to (dataset, fold) in `uci/results/task_map.csv`, never
under `~/ws/`. No path visible to an agent may contain `uci`, `redo`, a dataset name or
`cdtree`.

### 2.2 Launching an agent

Use the headless runner `adr.agent.run_agent` (`claude -p`):

```
run_agent(workdir, backend="claude", model=<model id>, budget_min=<10|60>, settings=<sandbox settings file>)
```

It works as follows:
- **Tools:** `--tools Bash Read Write Edit Glob Grep` restricts the session to these six tools.
  `--allowedTools` alone would only pre-approve them.
- **Disallowed:** `WebFetch`, `WebSearch`, `Agent` and `Task` are explicitly disallowed.
- **No extensions:** `--strict-mcp-config` loads no MCP servers or connectors, and
  `--disable-slash-commands` loads no skills.
- **Settings:** `settings` is passed as `--settings`, for the network sandbox of §2.3.
- **Deadline:** it enforces the wall-clock deadline.
- **Errors:** it resumes after harness errors while budget remains.
- **Logs:** it writes `agent_stream.jsonl` (full transcript, including input and output
  tokens) and `run_summary.json`.

Do not use in-session sub-agents. (An in-session sub-agent would receive the operator's memory index in its context.)

**Probe before any real run.** Launch `claude -p` with exactly `run_agent`'s flags in an empty
scratch directory, and ask it to list its tools, MCP servers and skills. Then check its stream:
- **Tools:** exactly the six tools.
- **Extensions:** no connectors, no skills, no `ListAgents`/`SendMessage`.
- **Context:** no operator memory, `CLAUDE.md` or project instructions. Grep the stream for
  `CDTree`, `UCI`, `Yang`, `benchmark` and `redo`.

If anything is wrong, stop and report to the operator.

**Prompt** (verbatim, with only the path substituted; it is `adr.agent.PROMPT`):

> You are a statistician solving a conditional density estimation task. Your working directory
> is `<task dir>`. Read TASK.md there and follow it exactly: it describes the data, the output
> contract, the rules (work only inside that directory, OMP_NUM_THREADS=1, at most 2 processes)
> and the deliverables (model.py passing the contract check, and notes.md ending with an
> `expected_nll: <value>` line). Take the time you need to explore the data, compare candidate
> programs with honest out-of-sample validation, and refine. When done, reply with a short
> summary: the final program's structure, number of constants, your validation NLL and
> expected_nll.

**What is scored** is the `model.py` on disk at the deadline, or when the agent stops if that
comes earlier. Copy it immediately to the results tree, together with `notes.md` and the
transcript.

### 2.3 Isolation and scheduling

- **Sandbox: enforced by the operating system, not only by rules.** Every agent runs with
  Claude Code's Bash sandbox (bubblewrap), configured by its own settings file
  (`~/ws/settings/<config>_<task>.json`). `wave.py` renders these files from the template installed by `setup.sh`, with the
  **absolute** path of the agent's task directory. A relative path such as `"."` would resolve
  against the settings file's folder, not the task directory. The sandbox:
  - denies all network access;
  - allows writes only in the task directory;
  - denies reads of `$HOME` and `/tmp`, except the task directory and the Python environment.

  Each agent also gets a private harness temp directory (`CLAUDE_CODE_TMPDIR`), so that the
  shared `/tmp/claude-<uid>`, which holds other sessions' logs, is not visible.
  `scripts/probe.py` verifies all of this with marker files before any real run.
- **Lock during any run:** `wave.py run` makes the whole package folder unreadable
  (`chmod 000`) while agents run, and unlocks it afterwards. This covers the README, the data,
  the results and the archived transcripts; its log lines are buffered meanwhile. The sandbox
  also denies reads of `$HOME` outside the task directory and of other task directories.
- **Leftover processes:** before starting a new batch, kill every leftover process whose
  working directory is under `~/ws/runs/`. Kill by PID and cwd, never with `pkill -f` (the pattern
  matches the calling shell).
- **Folds share data:** the training rows of fold k contain the test rows of fold k′. At any
  time, at most **one fold per dataset** may be on disk. The two configs may run the same
  fold at the same time.
- **Capacity:** each agent uses at most 2 processes with `OMP_NUM_THREADS=1`. A wave has 24
  tasks (12 datasets × 2 configs), and `wave.py run K` runs them all at once by default
  (`--parallel 28`, more than the 24 tasks); this needs at least 48 physical cores. On a smaller machine, or if the API
  returns rate-limit errors, lower `--parallel` and tell the operator.

### 2.4 Audit (after every batch, before archiving)

Run `audit.py <runs dir> --streams`. It scans every transcript for:
- web-tool calls;
- network commands;
- any path outside the task directory.

Review every flag manually and record a verdict for every run in `audit_notes.md`.

- **Integrity breach → the run is discarded and repeated**, however minor. This covers web or
  network access, or reading, listing or searching anything outside the task directory. It
  includes `pip list`, `ls site-packages`, `find` in `/tmp` or `$HOME`, and reading the
  project's copy of the library. With the sandbox, such attempts should fail, but an attempt
  is still a breach.
- **Conduct slip → recorded and kept.** This covers a third concurrent process, or scratch
  files written to `/tmp` with nothing read.
- **Allowed:** reading the log file of the agent's own background job at a path the harness
  gave it.

Keep discarded runs in `superseded/`, with the reason. Rerun them in a new directory (e.g.
`t0001r1`), never in the old one. Compliance counts per config are a reported result.

### 2.5 Contamination scan (at the end)

Run `contamination_scan.py` on all final transcripts. It searches only agent-written text for
benchmark, author and dataset names. Report the counts.

---

## 3. The experiment

### Data

The 14 files are in `data/uci/`. Verify `SHA256SUMS` before starting, and use the files
unchanged. The last column is the target; rows are already permuted and noise is already
added by the authors. In the paper's order, the datasets are:

energy, synchronous, localization, toxicity, concrete, slump, forestfires, NavalPropolsion,
sml2010, Thermography, support2, superconductivity.

SkillCraft and studentmath are **not analysed**: their targets are discrete (7 classes; grades on a
grid of thirds), for which a density NLL is not defined. This criterion is fixed before the runs
and follows Franco & Tedesco (arXiv:2511.14455, Table 3), who exclude the same two datasets. Their
files stay in `data/` unchanged (checksums) but are never used.

### Folds

There are 5 contiguous blocks. Fold k's test rows are the rows with
`rep(1:5, each = ceiling(n/5))[1:n] == k`, as in the authors' `run_cde.R` and in
`scripts/common.py`.

### Runs

12 datasets × 5 folds × 2 configs = **120 runs**.

### Score

The score is the **density test NLL**, the mean of `-log p(y | x)` over the test rows of the
fold, on the data exactly as released (with the noise added by the CDTree authors). It is the
score of the published columns. The agents are told this score in `TASK.md` (`adr.agent`,
`DEFAULT_JUDGED`) and nothing about the recording precision of the targets. Report mean
(sd, ddof = 1) over the 5 folds. A program that fails the contract, or is missing, scores nothing
and is reported as such. For every run, also record:
- the precision-aware NLL of `scripts/precision_score.py` (secondary diagnostic, never given to the agents);
- the number of constants;
- the optimism gap: test NLL minus the agent's `expected_nll`;
- the train NLL;
- wall minutes, turns, tool calls, and input and output tokens.

### Discreteness flags (diagnostic only)

Flag a program with † if either condition holds on 20 test rows:
- **lattice:** its density has a median of more than 10 local modes;
- **spike:** it has a peak narrower than 7 jitter standard deviations at the same y in at
  least 50% of the rows. The jitter is 0.01 · sd(y_train).

This is implemented in `scripts/score.py`. The flags are reported but no longer change any score.

### Table

- **Baseline columns:** the 10 baseline columns are copied from `published/table3_nll.csv`:
  CADET, CART-h, CART-k, CKDE, LSCDE, NKDE, CDTree, LinCDE, MDN, NF; an 11th, CPFN, from
  `published/cpfn_uci.csv` (Franco & Tedesco, Table 4; no standard deviations). All are ranked.
- **Agent columns:** add 2 shaded columns, one per config, with the density NLL; the
  precision-aware NLL of the same programs goes in `table_uci_precision.csv` (diagnostic).
- **Formatting:** bold marks the best value in each row; ‡ marks the lattice datasets
  (`lattice` in `data/uci_precision.csv`), whose targets lie on a lattice under the added noise;
  there a program that models the noise gains on the density NLL, for the agents and for the
  published methods alike.
- **Extra columns:** † flags, and the median number of constants per config.
- **Summary rows:** the average rank among all columns, over all datasets and without ‡.
- **Companion tables:**
  - cost per config: minutes, turns, tokens;
  - compliance per config: discarded runs and conduct slips;
  - self-assessment per config: optimism gap.

---

## 4. Replication package (what must exist at the end)

```
uci/
  README.md                  this file
  adr_snapshot/              the frozen library given to every agent
  sandbox_settings.json      the settings file passed to every agent (§2.3)
  data/uci/                  the 14 datasets, unchanged, with SHA256SUMS
  published/                 Table 3 of the paper, transcribed
  results/
    <config>/<dataset>_f<k>/
      model.py  notes.md  TASK.md  run_summary.json  agent_stream.jsonl.gz
    superseded/              discarded runs, each with the reason
    task_map.csv             task number -> (config, dataset, fold)
    audit_notes.md           a verdict for every run, compliance counts, every deviation
    usage.csv                per run: minutes, turns, tool calls, input and output tokens, attempts
  scripts/                   the pipeline: score.py -> results/per_fold.csv, make_tables.py -> results/tables/
```

The live task directories (`~/ws/runs/`) and the environment (`~/ws/env/`) are outside the
package and are not released. Every file that is kept from a run is copied into
`uci/results/`.

`reproduce.sh` must regenerate every table and number from `results/`, `data/uci/` and
`published/` alone, deterministically, in one command.

---

## 5. Reproducing the study

**Regenerate the tables** (no agent is called): `bash scripts/setup.sh && bash scripts/reproduce.sh`.
`reproduce.sh` scores the archived programs (`results/per_fold.csv`, `results/usage.csv`) and writes
`results/tables/`. It is deterministic.

**Re-run the agents.** All commands run from the package folder (`uci/`); `PY` stands for
`$ADR_WS/env/bin/python`, with `ADR_WS` set to a neutral workspace. If a script fails, report the error
and do not rewrite the pipeline.

1. **Setup:** `bash scripts/setup.sh`. It checks the tools (including that `bubblewrap` can create a
   sandbox; on Ubuntu 23.10+ run `sudo bash scripts/fix_apparmor.sh` once if needed), builds
   `$ADR_WS/env`, verifies the data checksums, freezes `adr_snapshot/` and installs the sandbox template.
2. **Harness probe:** `PY scripts/probe.py`. It runs one real agent session with the agents' exact flags and
   settings and checks from the tool results that there are exactly six tools, no connectors and no skills;
   curl and Python cannot reach the network; nothing outside the task directory can be read or written; the
   agents' Python works; and nothing identifying is in the context. All checks must print PASS.
3. **The experiment:** five waves, K = 1…5.
   ```
   PY scripts/wave.py prepare K        # 12 tasks: fold K of the 12 datasets
   nohup PY scripts/wave.py run K > $ADR_WS/waveK.log 2>&1 &   # about an hour; the package is locked meanwhile
   PY scripts/wave.py audit K          # after the run: flags -> results/audit_foldK.csv
   ```
   Always start `wave.py run` in the background. Poll `$ADR_WS/waveK.log`, which is outside the locked
   package; the run is over when it prints `done; leftovers killed`. Review every flag (§2.4). For each integrity
   breach: `PY scripts/wave.py rerun TASK --reason "..."`, run the wave again (only pending tasks run) and
   audit again. Then `PY scripts/wave.py archive K`. `wave.py status` shows progress.
4. **Closing:** `bash scripts/reproduce.sh`; `PY tools/contamination_scan.py` on the archived transcripts;
   the final audit counts per config.

Do not change the library, the contract, the prompt or the metrics once agents have started.
About $61 and 3 hours in total with 12 agents at once (five waves of 12 tasks).

---

## 6. Practical notes

- Use headless runs with the isolation flags, neutral paths, and the probe.
- Kill every leftover process before each batch, by PID or working directory, never with `pkill -f` (the
  pattern matches the calling shell); never reuse a task directory path.
- Headless streams include output tokens; report both input and output tokens.
- Scoring evaluates programs row by row and is slow: it runs in a process pool.
- Common integrity breaches of the smaller model: searching for or reading the project's copy of the library,
  listing installed packages, and `find` over temporary directories. Audit carefully.

---

## 7. What was run

Full audit record: `results/audit_notes.md`.

- **Runs.** 12 datasets × 5 folds = 60 runs (`opus-60`), with the covariate domain, run fold by fold with
  `--parallel 12` on the server described in the top-level README (workspace `/ws/a`), on 26 October 2026,
  with Claude Code 2.1.309.
- **Probe.** All 19 isolation checks PASS with Claude Opus 5.5 (`results/probe_*.json`).
- **Flags kept.** One false positive (a `sed` substitution containing `/opt`).
- **Scored.** All 60 programs pass the contract check with `--domain` and give finite densities at every
  test row.
- **Library.** The scorer of the precision-aware NLL is in `scripts/precision_score.py`, not in the library
  given to the agents; the agents were told the ordinary NLL and nothing about the recording precision.
- **Conduct.** No conduct slips.
- **Contamination scan.** No matches in the 60 transcripts (`results/contamination_scan.csv`).
- **Known limitations.** `run_summary.json` keeps the usage of the last `result` event only.
- **Cost.** About $61 (`results/usage.csv`).
- **Anonymisation.** The user name in file paths was replaced by `user` in every released file; nothing else
  was modified.
