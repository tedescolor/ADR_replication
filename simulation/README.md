# Agentic Distributional Regression: univariate simulation study

This file is the protocol of the study. It is also the set of instructions that the Claude Code
session orchestrating the runs received (`PROMPT.txt`): where it does not decide something, the
orchestrator asks the person running the study and never improvises a protocol choice.

**Goal.** Add rows for agentic distributional regression (ADR) to **Table 1** of Franco &
Tedesco, *Nonparametric estimation of conditional probability distributions using a generative
approach based on conditional push-forward neural networks* (arXiv:2511.14455, Section 4.1).
This is the univariate simulation study, here at sample sizes n = 250 and 1000; metrics AWD and
AQE(τ) for τ = 0.10, 0.25, 0.50, 0.75, 0.90. The analysis reproduces the authors' own code
**exactly** (`cpfn/branched.ipynb`): the same training sets (their seeds), the same true
quantiles (`qs_true.npy`) and the same metric computations. The published rows are copied
unchanged, and our rows are added. Two **agent configurations** are evaluated, both with a
60-minute budget:

| config | model | model id | budget |
|---|---|---|---|
| `sonnet-60` | Claude Sonnet 5.5 | `claude-sonnet-5-5` | 60 min |
| `opus-60` | Claude Opus 5.5 | `claude-opus-5-5` | 60 min |

**Replicates:** **all 100** of the authors' Monte Carlo replicates for n = 250 and n = 1000,
which are exactly their training sets 0–99 of those sizes: 2 × 100 × 2 = **400 agent runs**.
n = 500 is not analysed. The authors' seed counter still runs over n ∈ [250, 500, 1000]
(`AUTHOR_SIZES` in `scripts/common.py`), so replicate j of n = 1000 has seed 200 + j.

Agent runs cannot be reproduced token by token. The replication package therefore has two
parts: (i) every program, note and transcript, collected and released; (ii) every table and
number regenerated **deterministically from the released programs** by one script.

### Contents of this folder (self-contained; nothing has to be downloaded)

| path | what it is |
|---|---|
| `adr/` | the ADR library: `dist.py` (distribution families), `contract.py` (output checker), `agent.py` (task template, `prepare_task`, headless `run_agent`, `agent_flags`) |
| `cpfn/` | from github.com/NicolaRFranco/CPFN @ `6327b91`, unchanged (MIT license): `branched.ipynb` (the authors' data generation and metrics), `testcases/branched/qs_true.npy` (true conditional quantiles, 101 x-points × 99 levels), `SHA256SUMS`, `SOURCE.txt`, `LICENSE` |
| `published/table1.csv` | Table 1 of the paper, transcribed (verified number by number against the PDF), with `SOURCE.txt` |
| `scripts/` | **ready-to-run pipeline — use it, do not rewrite it**: `setup.sh`, `probe.py`, `validate.py`, `batch.py` (prepare / run / audit / archive / rerun / status), `score.py`, `score_exact.py`, `make_tables.py`, `reproduce.sh`, `fix_apparmor.sh` (operator, sudo), `common.py` (paths, configurations, and the line-by-line NumPy port of the authors' generator and metrics), `kcde_bandwidths.R` (bandwidths of the KCDE shown in the manuscript figures, R package `np`) |
| `sandbox_settings.json` | template of the sandbox settings given to every agent (installed by `setup.sh`) |
| `tools/` | `audit.py` (transcript audit, used by `batch.py audit`), `contamination_scan.py` |
| `requirements.txt` | pinned Python environment |

The machine needs Python 3.11, `bubblewrap` and `socat`, and the `claude` CLI logged in with
access to both models. Internet is needed only to install the pinned Python packages.

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

Check a program with:

```
python -m adr.contract model.py --features d --max-literals 80 --data data.csv
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

One task is one (sample size, replicate). The directory contains exactly these files:

- `data.csv`: the training set of that replicate, columns `x0` (the covariate X) and `y`.
- `adr/`: a copy of `adr_snapshot/`, excluding `agent.py`.
- `TASK.md`: rendered by `adr.agent.prepare_task` with the task's `n`, `d`, `K = 80`, the
  Python path and the **config's budget** (10 or 60).

**No test data exists on disk.** Test quantities (the true quantiles `qs_true.npy`, the test rows
for the NLL) are only in the locked package, or are generated at scoring time.

**Use neutral names.** Everything an agent can see — its task path, shown in the prompt, and
the Python path, shown in TASK.md — must be outside the package. Use one neutral workspace:

- **Python environment:** `$ADR_WS/env/`, so the agents call `$ADR_WS/env/bin/python`. Create it
  from `requirements.txt`.
- **Live task directories:** `$ADR_WS/runs/<config>/s001`, `s002`, …

Keep the mapping from task number to (sample size, replicate) in `results/task_map.csv`,
never under `$ADR_WS/`. No path visible to an agent may contain `redo`, `simulation`, `cpfn`,
`branched` or an author's name (`setup.sh` checks the workspace path).

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
- **Context:** no operator memory, `CLAUDE.md` or project instructions. `scripts/probe.py` greps the
  context for CPFN, Franco, push-forward, branched, simulation and redo.

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
  (`chmod 000`) while agents run, and unlocks it afterwards. This covers the README, the CPFN
  files, the results and the archived transcripts; its log lines are buffered meanwhile. The sandbox
  also denies reads of `$HOME` outside the task directory and of other task directories.
- **Leftover processes:** before starting a new batch, kill every leftover process whose
  working directory is under `$ADR_WS/runs/`. Kill by PID and cwd, never with `pkill -f` (the pattern
  matches the calling shell).
- **Scheduling:** the 200 replicates are independent datasets and no test data is on disk, so
  all 400 tasks can run in one batch.
- **Capacity:** each agent uses at most 2 processes with `OMP_NUM_THREADS=1`. Run at most one
  agent per 2 physical cores, keeping some for the system: `--parallel 28` (the default) needs
  56 cores. If the API returns rate-limit errors, lower `--parallel`.

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
`t0001r1`), never in the old one. Compliance counts per config are a reported result.

### 2.5 Contamination scan (at the end)

Run `contamination_scan.py` on all final transcripts. It searches only agent-written text for
benchmark and author names, and for the paper's name. Report the counts.

---

## 3. The experiment

### Data: exactly the authors' training sets

The model (their eq. 10; `f` and `true_sample` in `branched.ipynb`):
- `X ~ U(0, 1)`, `Z ~ N(0, 1)` and `P ~ U(0, 1)`, all independent.
- If `X < 0.5` or `P < 0.5`: `Y = 10X(X − 0.5)(1.5 − X) + 0.3 Z (1.3 − X)`.
- Otherwise: `Y = 10X(X − 0.5)(0.8 − X) + 0.3 Z (1.3 − X)`.

The notebook's `MCrun(ntrain, seed)` generates a training set as `np.random.seed(seed)`,
`xs = np.random.rand(n)`, then for each x in order `true_sample(x)` (one `randn()`, then one
`rand()`). The seed counter runs over `n ∈ [250, 500, 1000]` and `j ∈ range(100)`, so replicate
j uses seed j for n = 250, 100 + j for n = 500, and 200 + j for n = 1000.
`scripts/common.py:training_set(n, j)` reproduces this exactly; `validate.py` checks it
against `qs_true.npy`.

### Metrics: exactly the authors' code

These are `fastl1l1q_error` and `q_error` in `branched.ipynb`, ported line by line in
`scripts/common.py`.
- **Grids:** `xgrid = linspace(0, 1, 101)`, `qgrid = linspace(0, 1, 101)[1:-1]`.
- **Predicted quantiles:** at each `x` in `xgrid`, draw 1000 samples from the program
  (`model(x).sample(rng, 1000)`, the counterpart of `cpfn.sample(x, m = 1000)`), then take the
  empirical quantiles at `qgrid`, with linear interpolation as `torch.quantile` does.
- **AWD** = Simpson (dx = 0.01) over the levels of |true − predicted quantile|, then Simpson
  (dx = 0.01) over x. The true quantiles are `qs_true.npy`.
- **AQE(τ)** = Simpson (dx = 0.01) over x of |true − predicted quantile| at level τ. The level is
  located exactly on `qgrid`; the notebook's `argmax(qgrid == τ)` would silently return 0 if it
  were not, so the port asserts it.
- **Summary:** mean and sd (ddof = 1) over the replicates, as in the notebook.
- **Sampling seed:** the seed for sampling each program is fixed (`score.py`), so the scores
  are deterministic.

**Additional (ours, reported separately):**
- test NLL and **excess NLL** = NLL(program) − NLL(true density) on 10,000 independent rows
  (seed 900000 + training seed). It estimates the expected KL divergence to the truth.
- the number of constants;
- the optimism gap (test NLL minus the agent's own `expected_nll`);
- time, turns and tokens.

The agents are asked to minimise held-out NLL (the standard `TASK.md`), not AWD. ADR is a
likelihood-based estimator; AWD and AQE evaluate the distribution it returns.

### Table

`make_tables.py` writes:
- `results/tables/table1.csv` / `.tex`: the published rows of Table 1 unchanged (CPFN, np.kde,
  linear QR, qeML QR, fields QR, qrf QR; 100 replicates), then one row block per agent
  configuration (n = 250, 1000; 100 replicates), each cell mean (sd);
- companion tables: excess NLL, constants, self-assessment and cost.

---

## 4. Replication package (what must exist at the end)

```
simulation/
  README.md, PROMPT.txt, adr/, cpfn/, published/, scripts/, tools/, requirements.txt
  adr_snapshot/              the frozen library given to every agent
  sandbox_settings.json      the sandbox template given to every agent
  results/
    <config>/n<n>_r<rep>/    model.py  notes.md  TASK.md  run_summary.json  agent_stream.jsonl.gz
    superseded/              discarded runs, each with the reason
    task_map.csv             task number -> (config, n, replicate)
    audit_notes.md           a verdict for every run, compliance counts, every deviation
    validation.json, per_rep.csv, tables/
```

The live task directories (`$ADR_WS/runs/`) and the environment (`$ADR_WS/env/`) are outside the
package and are not released. `bash scripts/reproduce.sh` regenerates every table and number
from `results/`, `cpfn/` and `published/` alone.

---

## 5. Reproducing the study

**Regenerate the tables** (no agent is called): `bash scripts/setup.sh && bash scripts/reproduce.sh`.
`reproduce.sh` scores the archived programs (`results/per_rep.csv`) and writes `results/tables/`. It also scores
them with exact quantiles in place of Monte Carlo ones (`scripts/score_exact.py` -> `results/per_rep_exact.csv`;
manuscript, Appendix B.2).

**Re-run the agents.** Run every command from outside the package, e.g. from its parent folder, with paths such
as `PY simulation/scripts/batch.py ...`: while agents run, the package folder is unreadable and a shell whose
working directory is inside it cannot run any command. `PY` stands for `$ADR_WS/env/bin/python`, with `ADR_WS`
set to the neutral workspace. If a script fails, report the error and do not rewrite the pipeline.

1. **Setup:** `bash simulation/scripts/setup.sh` (tools, `bubblewrap` check, environment, CPFN checksums,
   `adr_snapshot/`, sandbox template; on Ubuntu 23.10+ `sudo bash scripts/fix_apparmor.sh` once if needed).
2. **Harness probe:** `PY simulation/scripts/probe.py`: one real agent session with the agents' exact flags; it
   checks six tools only, no network, no read or write outside the task directory, a working Python and no
   identifying words in the context. All checks must print PASS.
3. **Validation of the metrics:** `PY simulation/scripts/validate.py`: every quantile level is on the grid; the
   true model, written as an ADR program, scores AWD at the Monte Carlo floor (about 0.018) and an excess NLL
   of 0; `qs_true.npy` agrees with quantiles recomputed from the ported generator; a wrong program scores much
   worse; training sets are deterministic. All checks must print PASS.
4. **The experiment:** 400 runs in one batch.
   ```
   PY simulation/scripts/batch.py prepare
   nohup PY simulation/scripts/batch.py run --parallel P > $ADR_WS/batch.log 2>&1 &   # package locked meanwhile
   PY simulation/scripts/batch.py audit                      # after the run has finished
   ```
   Always start `batch.py run` in the background and poll `$ADR_WS/batch.log`; the run is over when it prints
   `done; leftovers killed`. Review every audit flag (§2.4). For each integrity breach:
   `PY simulation/scripts/batch.py rerun TASK --reason "..."`, run again (only pending tasks run) and audit again.
   Then `PY simulation/scripts/batch.py archive`. `batch.py status` shows progress.
5. **Closing:** `bash simulation/scripts/reproduce.sh`; `PY simulation/tools/contamination_scan.py` on the
   archived transcripts; the final audit counts per config.

Do not change the library, the contract, the prompt or the metrics once agents have started. With P agents in
parallel the batch takes about 400 × 8 / P minutes plus the longest run (at most 60 minutes); the cost is about
$0.3 per Sonnet run and $0.65 per Opus run.

---

## 6. Practical notes

- Use headless runs with the isolation flags, neutral paths, and the probe.
- Kill every leftover process before each batch, by PID or working directory, never with `pkill -f`; never reuse
  a task directory path.
- Headless streams include output tokens; report both input and output tokens.
- Scoring evaluates programs row by row: it runs in a process pool.
- Common integrity breaches of the smaller model: searching for or reading the project's copy of the library,
  listing installed packages, and `find` over temporary directories. Audit carefully.

---

## 7. What was run

Full audit record: `results/audit_notes.md`.

- **Runs.** `sonnet-60` and `opus-60`, 200 runs each (n = 250 and 1000, 100 replicates): 400 kept runs,
  `--parallel 36`. The orchestrating session was Claude Opus 5.5.
- **Reruns for agent conduct or output.** Opus: 0. Sonnet: 1 (s154, the run ended without delivering a program). Discarded runs are not released.
- **Compliance of kept runs.** No integrity breach. Conduct slips (a third concurrent process or a second
  Python interpreter): sonnet 11, opus 7.
- **Library exposure (known, small).** The snapshot given to the agents contained `adr/score.py`, the
  precision-aware scorer used in the UCI study, whose docstring says that structure finer than the recording
  precision earns nothing. `TASK.md` did not mention it (the task states the ordinary NLL). The text of that file
  appears in 40 of the 400 transcripts (the agent displayed it). Their scores do not differ from the others:
  opus n = 250, AWD 0.075 (12 runs) vs 0.076 (88); opus n = 1000, 0.044 (27) vs 0.042 (73); sonnet, 1 run
  exposed (0.160 vs 0.151). The file is not in the library of the other studies of this repository.
- **Contamination scan.** `tools/contamination_scan.py` searches agent-written text for CPFN, push-forward,
  the authors, KCDE, the arXiv id, `qs_true` and `branched`: 0 matches in the 400 transcripts
  (`results/contamination_scan.csv`).
- **Known limitations.** The cost table is computed from tokens and counts resumed sessions; the `turns` field is
  unreliable for some runs.
