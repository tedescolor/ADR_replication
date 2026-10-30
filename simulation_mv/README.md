# Agentic Distributional Regression: multivariate test case

This file is the protocol of the study. It is also the set of instructions that the Claude Code
session orchestrating the runs received (`PROMPT.txt`): where it does not decide something, the
orchestrator asks the person running the study and never improvises a protocol choice.

**Goal.** Add rows for agentic distributional regression (ADR) to **Table 2** of Franco &
Tedesco, *Nonparametric estimation of conditional probability distributions using a generative
approach based on conditional push-forward neural networks* (arXiv:2511.14455, Section 4.2): the
multivariate test case, a bivariate response `Y` and five covariates `X ~ N(0, I5)`, whose
conditional law is a mixture of a ring and two Gaussian blobs. The published rows (CPFN and KCDE)
are copied unchanged from the paper; they are **a single run** (one training set of n = 5000, one
fixed test set, no replicates, no standard deviations). We do the same: the authors' own training
set and the authors' own test set, **one agent run per configuration**. Two agent configurations
are evaluated, both with a 60-minute budget:

| config | model | model id | budget |
|---|---|---|---|
| `sonnet-60` | Claude Sonnet 5.5 | `claude-sonnet-5-5` | 60 min |
| `opus-60` | Claude Opus 5.5 | `claude-opus-5-5` | 60 min |

`REPS = 1` in `scripts/common.py` is the number of runs per config (2 agent runs in total). Agent
runs are stochastic: with a single run per config the agent rows carry the same kind of
uncertainty as the published rows, which also come from one fit. More runs on the same data
(`REPS` larger) change nothing else in the protocol.

Agent runs cannot be reproduced token by token. The replication package therefore has two parts:
(i) every program, note and transcript, collected and released; (ii) every table and number
regenerated **deterministically from the released programs** by one script.

### Contents of this folder (self-contained; nothing has to be downloaded)

| path | what it is |
|---|---|
| `adr/` | the ADR library: `dist.py` (distribution families, bivariate constructors), `contract.py` (output checker, `--ydim 2`), `agent.py` (task templates, `prepare_task`, headless `run_agent`, `agent_flags`) |
| `data/` | from github.com/NicolaRFranco/CPFN @ `6327b91` (MIT, `LICENSE_CPFN`): `blobs_train.csv` (the first 5000 rows of `blobs_data.csv`, the training set of CPFN and KCDE), `blobs_test.npy` (the fixed test set: 30 covariates × 1000 true samples), `blobs_ckde.npy` (the authors' KCDE samples, only used by `validate.py`), `blobs_3ex_true.csv` (true samples at the three covariates of Table 2, regenerated with the authors' R code by `make_3ex.R`), `SHA256SUMS`, `SOURCE.txt` |
| `published/table2.csv` | Table 2 of the paper, transcribed, with `SOURCE.txt` |
| `scripts/` | **ready-to-run pipeline — use it, do not rewrite it**: `setup.sh`, `probe.py`, `validate.py`, `batch.py` (prepare / run / audit / archive / rerun / status), `score.py`, `make_tables.py`, `reproduce.sh`, `fix_apparmor.sh` (operator, sudo), `common.py` (paths, configurations, the authors' metric and the true density) |
| `sandbox_settings.json` | template of the sandbox settings given to every agent (installed by `setup.sh`) |
| `tools/` | `audit.py` (transcript audit, used by `batch.py audit`), `contamination_scan.py` |
| `requirements.txt` | pinned Python environment (adds POT, used by the authors' metric) |

The machine needs Python 3.11, `bubblewrap` and `socat`, and the `claude` CLI logged in with
access to both models. Internet is needed only to install the pinned Python packages. R with
`mvtnorm` is needed only to regenerate `data/blobs_3ex_true.csv` (`Rscript data/make_3ex.R`).

---

## 1. The method and the estimator class (ADR)

An agent receives training data `(x_i, y_i)` and returns a Python program `model.py` that
defines `model(x)`. It maps one covariate vector `x` (a 1-d numpy array of length 5) to a **distribution
object** for the bivariate response `y = (y0, y1)`. The program is the estimate `p̂(y | x)`. It must be a proper conditional
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
- **Multivariate constructors:** the response is bivariate, so these are the building blocks of
  every program: `MultivariateNormal(mean, cov)`; `Independent(d1, d2)`; `Sequential(first, cond)`,
  that is `p(y1) · p(y2 | y1)` with `cond(y1)` returning the distribution of `y2`; and `Mixture` of
  any of them (all components of the same dimension). `Sequential` and `Mixture` can be nested.
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
  `Distribution` of dimension 2 whose samples have finite log-density.

Check a program with:

```
python -m adr.contract model.py --features 5 --ydim 2 --max-literals 80 --data data.csv
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

One task is one agent run (a config and a run index). The directory contains exactly these files:

- `data.csv`: the authors' training set (5000 rows), columns `x0 … x4` (the covariates) and `y0`, `y1`.
- `adr/`: a copy of `adr_snapshot/`, excluding `agent.py`.
- `TASK.md`: rendered by `adr.agent.prepare_task` with the task's `n`, `d`, `K = 80`, the
  Python path and the **config's budget** (60).

**No test data is in any task directory.** The fixed test set (`data/blobs_test.npy`) and the other
evaluation files are only in the locked package.

**Use neutral names.** Everything an agent can see — its task path, shown in the prompt, and
the Python path, shown in TASK.md — must be outside the package. Use one neutral workspace:

- **Python environment:** `$ADR_WS/env/`, so the agents call `$ADR_WS/env/bin/python`. Create it
  from `requirements.txt`.
- **Live task directories:** `$ADR_WS/runs/<config>/s001`, `s002`, …

Keep the mapping from task number to (config, run) in `results/task_map.csv`,
never under `$ADR_WS/`. No path visible to an agent may contain `redo`, `simulation`, `multiv`, `cpfn`,
`blobs` or an author's name (`setup.sh` checks the workspace path).

**Workspace for this analysis: `$ADR_WS`**, a neutral directory outside the home directory
(for example `/ws`). The operator creates it and exports `ADR_WS` before starting (see the top-level README),
because the user name in `/home/<user>/` may contain an author's name. Keep `ADR_WS` set on
every command; `setup.sh` refuses a workspace path containing an
identifying word. The home path still appears inside the harness itself: in the sandbox
rule that denies reads of `$HOME`, and in environment variables the agents are not shown. This
is an **accepted, documented limitation**; do not stop for it, record it once in
`audit_notes.md`.

### 2.2 Launching an agent

Use the headless runner `adr.agent.run_agent` (`claude -p`):

```
run_agent(workdir, backend="claude", model=<model id>, budget_min=60, settings=<sandbox settings file>)
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
- **Context:** no operator memory, `CLAUDE.md` or project instructions. `scripts/probe.py` greps the
  context for CPFN, Franco, push-forward, branched, simulation, KCDE, blobs and redo.

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
  (`$ADR_WS/settings/<config>_<task>.json`). `batch.py` renders these files from the template installed by `setup.sh`, with the
  **absolute** path of the agent's task directory. A relative path such as `"."` would resolve
  against the settings file's folder, not the task directory. The sandbox:
  - denies all network access;
  - allows writes only in the task directory;
  - denies reads of `$HOME` and `/tmp`, except the task directory and the Python environment.

  Each agent also gets a private harness temp directory (`CLAUDE_CODE_TMPDIR`), so that the
  shared `/tmp/claude-<uid>`, which holds other sessions' logs, is not visible.
  `scripts/probe.py` verifies all of this with marker files before any real run.
- **Lock during any run:** `batch.py run` makes the whole package folder unreadable
  (`chmod 000`) while agents run, and unlocks it afterwards. This covers the README, the data
  files, the results and the archived transcripts; its log lines are buffered meanwhile. The sandbox
  also denies reads of `$HOME` outside the task directory and of other task directories.
- **Leftover processes:** before starting a new batch, kill every leftover process whose
  working directory is under `$ADR_WS/runs/`. Kill by PID and cwd, never with `pkill -f` (the pattern
  matches the calling shell).
- **Scheduling:** all tasks use the same training data and no test data is on disk, so all runs
  can go in one batch.
- **Capacity:** each agent uses at most 2 processes with `OMP_NUM_THREADS=1`; all runs of a
  batch go at once (`--parallel` defaults to 2 per config). If the API returns
  rate-limit errors, stop, lower `--parallel` and tell the operator.

### 2.4 Audit (after the batch, before archiving)

`PY scripts/batch.py audit` scans every transcript for:
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
`s001r1`), never in the old one. Compliance counts per config are a reported result.

### 2.5 Contamination scan (at the end)

Run `contamination_scan.py` on all final transcripts. It searches only agent-written text for
benchmark and author names, and for the paper's name. Report the counts.

---

## 3. The experiment

### Data: exactly the authors' files
- **Training set:** the first 5000 rows of `blobs_data.csv` (`data/blobs_train.csv`): covariates
  `X1..X5` become `x0..x4`, responses `Y1, Y2` become `y0, y1`. This is what CPFN
  (`blobs.ipynb`: `x[:ntrain]`, `ntrain = 5000`) and KCDE (`CKDE_multivariate.R`: `train[1:5000, ]`)
  were fitted on.
- **Test set:** `blobs_test.npy`, shape (30, 1000, 7): 30 covariate vectors drawn from `N(0, I5)`
  (`set.seed(123)`) with 1000 samples of the true `Y | X = x` each. It is only used at scoring time.
- **Three covariates of Table 2:** `x_ring = (2, 1, .5, -.3, -1)`, `x_trans = 0`,
  `x_blobs = (-2, -1, -.5, .3, 1)` (as in the authors' R code). Their true samples
  (`set.seed(1234)`) are not in the authors' repository; `data/make_3ex.R` regenerates them with the
  authors' own functions, and the same script reproduces `blobs_test.npy` to 1e-14.

### Metric: the authors' code
`AWD` is `avgW1error` of `generative.py`: for each of the 30 test covariates, draw 1000 samples
from the estimator, compute the 1-Wasserstein distance (POT `emd2`, Euclidean cost, uniform weights)
to the 1000 true samples, and average over the 30 covariates. For an ADR program the samples come
from `model(x).sample(rng, 1000)` with a fixed seed. The three W1 columns are the same distance at
the three covariates above. Secondary, ours and not in the paper: the excess NLL over the true
density on the 30 × 1000 true test samples (`true_logpdf` in `scripts/common.py`, a port of the
authors' R density), the number of constants, the optimism gap, and the cost. The agents are asked
to minimise held-out NLL (the standard `TASK.md`), not AWD.

**The floor.** With 1000 samples on each side the AWD of a perfect estimator is not 0:
`validate.py` measures it (about 0.17) by comparing fresh samples of the true model with the
authors' true samples. It is reported as a row of the table.

### Table
- **Published rows:** CPFN and KCDE, copied unchanged from `published/table2.csv`.
- **Agent rows:** one per config; AWD and the three W1 columns of the scored run.
- **Floor row:** the true model against the authors' true samples.
- **Companion tables:** excess NLL, constants, optimism gap, cost.

### What is known about the original table (read before interpreting)
- Table 2 is one run: the CPFN vs KCDE gap is not accompanied by any variability estimate.
- `blobs.ipynb` reproduces the AWD column (CPFN 0.2784, KCDE 0.6843; `validate.py` recomputes the
  KCDE value). It does **not** reproduce the three W1 columns: it computes W1 at test indices
  0, 5, 10 (covariates drawn at random), not at `x_ring`, `x_trans`, `x_blobs`, and its outputs there
  differ from the printed columns. The code that produced the printed W1 columns is not in the
  repository. Our W1 columns use the three covariates named in the paper's R code, and are not
  guaranteed to match the construction behind the printed ones.
- In the paper's text the vectors `x_ring` and `x_blobs` are printed identically (a typo); the
  authors' R code has the vectors above.
- `rot2` in the authors' R code fills its matrix by column, so it rotates by `-theta`; a generator
  written from the paper's formulas alone would be wrong. `common.py` follows the R code.

---

## 4. Replication package (what must exist at the end)

```
simulation_mv/
  README.md  adr_snapshot/  sandbox_settings.json  data/  published/  scripts/  tools/
  results/
    <config>/run<k>/model.py  notes.md  TASK.md  run_summary.json  agent_stream.jsonl.gz  task.json
    superseded/              discarded runs, each with the reason (not released)
    task_map.csv  audit.csv  audit_notes.md  batch_log.txt  validation.json  contamination_scan.csv
    per_run.csv  per_x.csv  tables/
```

`reproduce.sh` regenerates every table and number from `results/`, `data/` and `published/` alone,
deterministically, in one command.

---

## 5. Reproducing the study

**Regenerate the table** (no agent is called): `bash scripts/setup.sh && bash scripts/reproduce.sh`
(validation, scoring of the archived programs, `results/tables/`). It is deterministic.

**Re-run the agents.** Run every command from outside the package (a shell whose working directory is inside it
cannot run commands while agents run). `PY` stands for `$ADR_WS/env/bin/python`, with `ADR_WS` set to the
neutral workspace.

1. **Setup:** `bash simulation_mv/scripts/setup.sh` (tools, `bubblewrap` check, environment with POT, data
   checksums, `adr_snapshot/`, sandbox template).
2. **Harness probe:** `PY simulation_mv/scripts/probe.py`; all checks must print PASS.
3. **Validation:** `PY simulation_mv/scripts/validate.py`; all checks must print PASS (it reports the floor of the
   metric and the AWD of a wrong program).
4. **Prepare and run:**
   ```
   PY simulation_mv/scripts/batch.py prepare
   PY simulation_mv/scripts/batch.py run          # two agents at once, up to 60 minutes
   ```
5. **Audit:** `PY simulation_mv/scripts/batch.py audit`; review every flag (§2.4); rerun integrity breaches with
   `batch.py rerun TASK --reason "..."`.
6. **Archive and score:** `PY simulation_mv/scripts/batch.py archive`, then `bash simulation_mv/scripts/reproduce.sh`.
7. **Contamination scan:** `PY simulation_mv/tools/contamination_scan.py results/*/run*/agent_stream.jsonl.gz --out results/contamination_scan.csv`.

Do not change the library, the contract, the prompt or the metrics once agents have started. About $1 per Sonnet run
and $1 to $3 per Opus run.

---

## 6. Practical notes

- Use headless runs with the isolation flags, neutral paths, and the probe.
- Kill leftover processes by PID or working directory, never with `pkill -f`; never reuse a task directory path.
- Never change the library once `adr_snapshot/` exists: it is what the agents get.
- Headless streams include output tokens; report both input and output tokens.

---

## 7. What was run

Full audit record: `results/audit_notes.md`.

- **Runs.** One run per config, as designed: `sonnet-60` (s001) and `opus-60` (s002), both kept, no rerun. Total cost $2.06.
- **Result (AWD, Table 2):** ADR Opus 5.5 **0.193**, ADR Sonnet 5.5 **0.789**; published CPFN 0.278 and KCDE 0.684;
  floor of the metric (true model, 1000 vs 1000 samples) 0.171. One run per row: no variability estimate, as in the
  published rows.
- **Compliance.** Both runs audited clean by `tools/audit.py` and by manual review of every tool call. Opus: one
  conduct slip (a third concurrent process for about 15 s). Sonnet: an attempt to call a tool it does not have
  (`Monitor`), `ps aux | grep` on its own job, and an import probe of torch, jax, autograd, sklearn and statsmodels;
  these were not treated as integrity breaches and the run was kept.
- **Stopping.** Both agents stopped on their own well inside the budget (opus 12.3 min, sonnet 33.3 min).
- **W1 columns.** They use the three covariates of the authors' R code and are not guaranteed to match the
  construction behind the printed W1 columns (§3); compare on AWD.
- **Contamination scan.** 0 matches in the 2 transcripts. Most reasoning blocks of the headless streams are recorded
  empty, so hidden reasoning is only partly searchable.
