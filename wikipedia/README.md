# Agentic Distributional Regression: a dataset created after the agents' training data

The UCI datasets of the main benchmark are public and old, so an agent may have met them, or
analyses of them, during pretraining. This study repeats the UCI protocol on a dataset that no
pretrained model can have seen: a random sample of English Wikipedia articles drawn with
`../wikipedia_dataset/generate_dataset.py`, whose two responses cover a period ending on the day
the script was run. The agents receive only the training rows, with neutral column names and no
description of the data.

## Contents

| path | what it is |
|---|---|
| `data/dataset.csv` | the file written by `generate_dataset.py` (identical to `../wikipedia_dataset/dataset.csv`): 1000 rows, covariates `x1`–`x6`, responses `y1`, `y2` |
| `data/dataset_dq.csv` | the same rows with dequantized responses (below); the only file used by the study |
| `data/SHA256SUMS` | checksums of both files |
| `adr/`, `adr_snapshot/` | the ADR library, identical to the one of the UCI and bivariate studies (`agent.py` has the bivariate task template of `../simulation_mv`) |
| `scripts/` | `make_dequantized.py`, `setup.sh`, `probe.py`, `batch.py` (agents), `score.py`, baselines, `make_tables.py` |
| `tools/` | `audit.py`, `contamination_scan.py` |
| `sandbox_settings.json` | the sandbox settings template given to every agent |
| `results/` | archived programs, notes, transcripts, audit records, scores and tables |

## Data

`dataset.csv` has six covariates (numbers of categories, internal links and interlanguage links,
a lead-image indicator, article age in years, wikitext size in KiB) and two responses,
`y1 = log(1 + page views over 30 days)` and `y2 = log(1 + edits over 12 months)`; see
`../wikipedia_dataset/README.md` for the sampling design. None of this is shown to the agents.

**Dequantization.** The responses are logarithms of counts, so their conditional law is discrete
and a density NLL is not defined: a density can place arbitrarily high spikes on the atoms
(`y2` has 43 distinct values and 18% zeros). As for the two discrete UCI datasets, the density
benchmark is not applicable to the raw file. Each response is therefore replaced, once and with a
fixed seed, by `log(1 + c + U)` with `U ~ Uniform(0, 1)` independent across rows and responses
(`scripts/make_dequantized.py`). The result has a density, and its density NLL equals the NLL of
the counts up to a term that does not depend on the estimator. Every method receives the same file.

## Protocol (as in the UCI study, `../uci/README.md`)

- **Folds:** five contiguous blocks of 200 rows in file order (`rep(1:5, each = 200)`).
- **Agent:** Claude Opus 5.5 (`opus-60`), 60-minute budget, one run per fold; same library,
  output contract (80 constants, 500 AST nodes), prompt, tools, sandbox and audit as the UCI study.
  The task directory contains only `data.csv` (the 800 training rows of the fold, columns
  `x0`–`x5`, `y0`, `y1`), the library, `TASK.md` (bivariate template) and `domain.csv`, the range of
  each covariate over the full dataset, which the contract check enforces as in the UCI study. Test rows are never
  written to a task directory, and at most one fold is on disk at any time.
- **Score:** mean test negative log-likelihood of the joint density of `(y1, y2)` on the
  dequantized data, per fold; reported as the mean over the five folds.
- **Competitors:** KCDE (R package `np` 0.70.5, `npcdens` with likelihood cross-validated
  bandwidths, `x4` as a binary factor; `scripts/baselines/kcde_fit.R`) and CPFN (Python package
  `cpfn` 1.0.5; `scripts/baselines/cpfn_fit.py`), fitted on the same folds. CPFN uses the settings of
  the CPFN paper's UCI study (width 50, three gelu layers, rank 50, 2000 epochs, 10% of the training
  rows held out to keep the best state, `val_split=0.1`) with R = 500 latent draws in the loss, on a
  GPU; the run with the paper's R = 100 (CPU) is kept as a diagnostic. Its test density uses 20000
  latent draws per row and is mapped back to the original scale (the package works on z-scores).

## Reproducing

Regenerate the scores and the table from the archived programs (no agent call):

```
bash scripts/setup.sh && bash scripts/reproduce.sh
```

Re-run the agents and the baselines:

```
bash scripts/setup.sh
PY scripts/probe.py --model claude-opus-5-5
PY scripts/batch.py cycle            # five folds, one after the other
Rscript scripts/baselines/kcde_fit.R K data/dataset_dq.csv results/baselines     # K = 1..5
python scripts/baselines/cpfn_fit.py --m 500 --device cuda --out results/baselines
```

## What was run

- **Probe:** all 19 isolation checks PASS with Claude Opus 5.5 (`results/probe_*.json`).
- **Workspace.** The agents' workspace was `/ws/b` on the server described in the top-level README; the probe
  verified that nothing outside the task directory could be read. The user name in file paths was replaced by
  `user` in every released file; nothing else was modified.
- **Agent runs:** five folds, one Opus run each (6.4-50.7 min, $0.53-1.42 per run), on 26 October 2026, with
  the covariate domain and Claude Code 2.1.309. Audit: all five runs clean. Contamination scan: one false
  positive (`categories`, a pandas attribute). Details: `results/audit_notes.md`.
- **Scores** (`scripts/score.py`, `scripts/make_tables.py` -> `results/tables/table_nll.csv`), test NLL;
  every fold is scored:

| | KCDE | CPFN | ADR: Opus 5.5 |
|---|---|---|---|
| fold 1 | 2.839 | 3.064 | 2.584 |
| fold 2 | 2.675 | 3.066 | 2.399 |
| fold 3 | 2.711 | 2.913 | 2.529 |
| fold 4 | 2.774 | 3.386 | 2.522 |
| fold 5 | 2.698 | 2.722 | 2.439 |
| mean | 2.739 | 3.030 | 2.495 |

CPFN with the paper's R = 100: 4.906, 4.782, 2.956, 4.192, 2.737 (mean 3.914).
