# Task: conditional density estimation

`data.csv` has 413 rows: 12 covariates `x0`..`x11` and a target `y`.
Write `model.py` defining `model(x)` that, for one covariate row `x` (a numpy array of
length 12, in the column order of `data.csv`), returns an `adr.dist.Distribution` for `y`.

**How you are judged.** The mean negative log-likelihood (natural log) of `model` on
held-out rows from the same distribution, which you do not have. Lower is better.
The density must be on the original scale of `y`: if you transform `y` while
exploring, the returned distribution must still be a density for the untransformed `y`.
Overfitting `data.csv` is your responsibility; how you validate is up to you.

**Contract** (read `adr/dist.py` and `adr/contract.py`):
- `model.py` imports only from `adr.dist` and `math`; `model` is a pure function with its
  constants written in as numbers;
- at most 80 constants and 500 AST nodes. Constants are counted as: every
  numeric or boolean literal and every use of `pi` or `e`; integer column indices such as
  the 3 in `x[3]` are not counted. String constants are not allowed (except docstrings);
- no loops, comprehensions, dicts or sets; no list/tuple literal longer than 6;
- check it with
  `python -m adr.contract model.py --features 12 --max-literals 80 --data data.csv --domain domain.csv`
  (a model that fails the check scores nothing).

**Covariate domain.** `domain.csv` gives, for each covariate, the smallest and the largest value
in the full dataset, including the held-out rows on which `model` is scored. Every row on which
`model` is evaluated lies in this box. `model(x)` must return a valid distribution for every `x`
in the box, not only for the rows of `data.csv`: the contract check also evaluates it at points
of the box, including each covariate at its minimum and at its maximum.

**Rules** (every action is logged and audited; a run that breaks them is discarded).
- Use only `data.csv` and your own analysis. No internet: do not use web search or fetch
  tools, do not download or install anything, do not call any network service.
- Work only inside this directory: put scratch files in `./work/`, not in /tmp or any
  scratch/temp directory, and do not read, list or search anything outside this directory.
- Do not try to identify the dataset or find other copies of it; treat it as anonymous.
- Use `/ws/a/env/bin/python` (numpy, scipy, pandas, matplotlib). Run with `OMP_NUM_THREADS=1` and
  at most 2 processes at a time: other agents share the machine.

**Time budget: 60 minutes** of wall-clock time from the start. Save a valid `model.py`
(and `notes.md`) early and overwrite them as you improve; at 60 minutes the run is stopped
and whatever `model.py` is on disk is scored.

**Deliverables** (in this directory):
- `model.py`, passing the check;
- `notes.md`: what you tried, the validation scheme, why you chose the final program, and
  your estimate of its mean NLL on new data (one number, on its own line as
  `expected_nll: <value>`).
