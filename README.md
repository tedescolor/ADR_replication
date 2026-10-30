# ADR: Agentic Distributional Regression, re-analysis

Code, agent programs and transcripts for four studies of *Agentic Distributional Regression*
(ADR), where an LLM agent writes a program `model.py` that returns a conditional distribution
`p(y | x)` (a program in the library `adr/`, checked by a contract: at most 80 constants and 500 AST nodes):

| folder | study | agent runs | configurations |
|---|---|---|---|
| `uci/` | UCI benchmark of Yang & van Leeuwen (NeurIPS 2024, Table 3): 12 continuous-target datasets x 5 folds, scored by the ordinary test NLL, the score of the published methods; the agent receives the range of every covariate over the full dataset | 60 | `opus-60` |
| `simulation/` | univariate simulation of Franco & Tedesco (arXiv:2511.14455, Table 1): n = 250 and 1000, 100 replicates | 400 | `sonnet-60`, `opus-60` |
| `simulation_mv/` | multivariate test case of the same paper (Table 2): bivariate response, the authors' own training and test data, one run per config | 2 | `sonnet-60`, `opus-60` |
| `wikipedia/` | a new dataset created for this study (`wikipedia_dataset/`): 1000 random English Wikipedia articles, 6 covariates, bivariate response (page views and edits) covering a period after the agents' training data; 5 folds as in `uci/`, compared with KCDE and CPFN | 5 | `opus-60` |

The simulation studies use Claude Sonnet 5.5 (`claude-sonnet-5-5`) and Claude Opus 5.5 (`claude-opus-5-5`); the
real-data studies (`uci/`, `wikipedia/`) use Claude Opus 5.5 only. Every run has a 60-minute budget and is run headless and sandboxed (no network). Each study has its own protocol in its
`README.md`; Section 7 of each lists what was actually run and every deviation from the protocol.

## Computing environment

All agent runs were carried out on one server: two AMD EPYC 7302 processors (32 cores, 64 threads),
503 GB of memory, Ubuntu 24.04. The agents were run headless with Claude Code (version 2.1.287 for the
univariate simulation, 2.1.288 for the first runs of the bivariate simulation, 2.1.309 for the repeated
bivariate runs, the UCI benchmark and the Wikipedia study), with 36, 2, 12 and 1 agents at a time in the
univariate, bivariate, UCI and Wikipedia studies.

## What can and cannot be replicated

Agent runs are not reproducible token by token. The package therefore has two parts:

1. **Released artefacts.** Every final program (`model.py`), the agent's notes, the task text, the run
   summary and the full transcript (`agent_stream.jsonl.gz`) of every kept run, under
   `<study>/results/<config>/<run>/`. Discarded runs are not included.
2. **Deterministic regeneration.** Every table and number is recomputed from these programs by one
   command per study (`reproduce.sh`). No agent is called.

Re-running the agents yourself would give different programs and, in general, different numbers.

## Regenerate the tables

```bash
export ADR_WS=/path/to/workspace        # a directory you own; the Python environment is built in $ADR_WS/env
bash uci/scripts/setup.sh && bash uci/scripts/reproduce.sh
bash simulation/scripts/setup.sh && bash simulation/scripts/reproduce.sh
bash simulation_mv/scripts/setup.sh && bash simulation_mv/scripts/reproduce.sh
bash wikipedia/scripts/setup.sh && bash wikipedia/scripts/reproduce.sh
```

The figures and table fragments of the manuscript are regenerated from the same archived results by
`generate_images/` (see its `README.txt`); no agent is called.

Python 3.11 (or `uv`) is needed. The three studies pin different environments (`requirements.txt`);
`setup.sh` of each study installs its own into `$ADR_WS/env`, so run the studies one after the other.
`setup.sh` also checks for `bubblewrap` and `socat`, which are needed only to re-run agents.

## Re-running the agents

Each study's `README.md` is the protocol and `PROMPT.txt` the prompt that was given to the Claude Code
session that orchestrated it. Requirements: the `claude` CLI logged in with access to the two models,
`bubblewrap` and `socat` (unprivileged user namespaces must work; `scripts/fix_apparmor.sh` is for Ubuntu 23.10+),
internet access to the Anthropic API, and a **neutral workspace outside the home directory** whose path contains
no user name or project word (`setup.sh` refuses an identifying path), for example `ADR_WS=/ws`. The studies share
the workspace and the CPUs: run them one at a time, each in its own Claude Code session started in the repository
folder, and empty the workspace between studies.

## Layout

```
uci/ , simulation/ , simulation_mv/ , wikipedia/
  README.md             protocol and Section 7 (what was actually run, deviations)
  PROMPT.txt            prompt of the orchestrating session
  adr/, adr_snapshot/   the ADR library and the frozen copy given to every agent
  data/ , cpfn/         input data (checksums included) / CPFN reference code
  published/            the published tables, transcribed
  scripts/ , tools/     the pipeline (setup, wave/batch, probe, score, make_tables, reproduce, audit)
  results/              programs, notes, transcripts, audits, per-run scores, tables
```

## Data and licences

- Code in this repository: MIT (`LICENSE`).
- `uci/data/` contains the UCI datasets as released by the CDTree authors (github.com/ylincen/CDTree,
  commit `b607e12`), unchanged, with `SHA256SUMS`. That repository declares no licence; the files are
  redistributed only to make replication possible and remain subject to the terms of the original UCI sources
  and of the CDTree release.
- `simulation/cpfn/` and `simulation_mv/data/` come from github.com/NicolaRFranco/CPFN (MIT), unchanged,
  except `simulation_mv/data/blobs_3ex_true.csv`, regenerated with the authors' own R code (`make_3ex.R`).
- `wikipedia_dataset/` contains the script that generated the new dataset and its output; the data come from
  the public MediaWiki and Wikimedia Pageviews APIs.
- Transcripts were anonymised: the user name in file paths was replaced by `user`. Nothing else
  in them was modified.

## Notes on interpretation

- UCI: all methods are compared with the same score, the ordinary test NLL. Agents were told this score and
  nothing about the recording precision of the targets. A precision-aware NLL is recorded as a secondary
  diagnostic only (`uci/results/tables/table_uci_precision.csv`).
- Multivariate case: the published Table 2 is a single run (one training set, one test set); the agent rows are
  single runs too. Our W1 columns at the three named covariates are not guaranteed to match the construction
  behind the printed ones (see `simulation_mv/README.md`, Section 3).
- The agents' prior knowledge of the domain (for example, recognising concrete mixtures from the covariates) is
  not detected by the contamination scan.
