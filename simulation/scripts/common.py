"""Shared definitions of the simulation re-analysis: paths, configurations, tasks, and a
line-by-line NumPy port of the CPFN authors' data generation and metrics (cpfn/branched.ipynb).

PKG  = this package. WS = neutral workspace seen by the agents (default ~/ws; ADR_WS overrides):
       WS/env (Python), WS/runs/<config>/sNNN (task dirs), WS/settings, WS/tmp.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.stats import norm

PKG = Path(__file__).resolve().parents[1]
WS = Path(os.environ.get("ADR_WS", "~/ws")).expanduser()
ENV_PY = WS / "env" / "bin" / "python"
RUNS = WS / "runs"
SNAPSHOT = PKG / "adr_snapshot"
RESULTS = PKG / "results"
SETTINGS_TEMPLATE = WS / "settings.template.json"
SETTINGS_DIR = WS / "settings"
TMP_DIR = WS / "tmp"
TASK_MAP = RESULTS / "task_map.csv"
Q_TRUE = PKG / "cpfn" / "testcases" / "branched" / "qs_true.npy"

CONFIGS = {  # name: (model id, budget in minutes)
    "sonnet-60": ("claude-sonnet-5-5", 60),
    "opus-60": ("claude-opus-5-5", 60),
}
AUTHOR_SIZES = [250, 500, 1000]   # the paper's sizes: they define the notebook's seed counter
SIZES = [250, 1000]               # the sizes analysed here (a subset of AUTHOR_SIZES)
REPLICATES = 100                  # all of the authors' Monte Carlo replicates
MAX_LITERALS = 80
QUANTILE_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]

sys.path.insert(0, str(PKG))      # `adr` package (identical to adr_snapshot)


# ---------------------------------------------------------------- data generation (branched.ipynb)
def f(x, z, p):
    if x < 0.5 or p < 0.5:
        return 10 * x * (x - 0.5) * (1.5 - x) + z * 0.3 * (1.3 - x)
    return 10 * x * (x - 0.5) * (0.8 - x) + z * 0.3 * (1.3 - x)


def true_sample(x):
    z = np.random.randn()
    p = np.random.rand()
    return f(x, z, p)


def training_seed(n, rep):
    """The notebook's seed counter: seed = 0.. over n in [250, 500, 1000], j in range(100).
    It is defined on AUTHOR_SIZES, not on SIZES: replicate j of n = 1000 has seed 200 + j even
    though n = 500 is not analysed here."""
    return AUTHOR_SIZES.index(n) * 100 + rep


def training_set(n, rep):
    """Exactly MCrun(ntrain=n, seed): np.random.seed(seed); xs = rand(n); ys = [true_sample(x)]."""
    np.random.seed(training_seed(n, rep))
    xs = np.random.rand(n)
    ys = np.array([true_sample(x) for x in xs])
    return xs, ys


# ---------------------------------------------------------------- metrics (branched.ipynb)
XGRID = np.linspace(0, 1, 101)
QGRID = np.linspace(0, 1, 101)[1:-1]


def q_index(q0):
    """The notebook uses np.argmax(qgrid == q0), which silently returns 0 if no level matches."""
    j = int(np.argmax(QGRID == q0))
    assert QGRID[j] == q0, f"quantile level {q0} not on the grid"
    return j


def quantile_errors(model, rng, m=1000):
    """|Q_true - Q_pred| on the 101 x 99 grid; Q_pred from m samples of the program at each x
    (the notebook: cpfn.sample(x, m=1000), then torch quantile = linear interpolation)."""
    Q = np.load(Q_TRUE)
    err = np.empty_like(Q)
    for k, x in enumerate(XGRID):
        yg = np.asarray(model(np.array([x])).sample(rng, m), dtype=float).ravel()
        err[k] = np.abs(Q[k] - np.quantile(yg, QGRID))
    return err


def awd(err):
    """fastl1l1q_error: Simpson (dx = 0.01) over the quantile levels, then over x."""
    return simpson(np.abs([simpson(e, dx=1e-2) for e in err]), dx=1e-2)


def aqe(err, q0):
    """q_error: Simpson (dx = 0.01) over x of |q_true - q_pred| at level q0."""
    return simpson(np.abs(err[:, q_index(q0)]), dx=1e-2)


# ---------------------------------------------------------------- true density (branched.ipynb)
def true_logpdf(y, x):
    e1 = 10 * x * (x - 0.5) * (1.5 - x)
    e2 = 10 * x * (x - 0.5) * (0.8 - x)
    s = np.abs(0.3 * (1.3 - x))
    one = norm.logpdf(y, e1, s)
    two = np.logaddexp(norm.logpdf(y, e1, s), norm.logpdf(y, e2, s)) + np.log(0.5)
    return np.where(x < 0.5, one, two)


def test_set(n, rep, size=10_000):
    """Independent test rows for the excess NLL (ours; not in the paper)."""
    np.random.seed(900_000 + training_seed(n, rep))
    xs = np.random.rand(size)
    ys = np.array([true_sample(x) for x in xs])
    return xs, ys


# ---------------------------------------------------------------- tasks
def task_number(config, n, rep):
    """Neutral task number s001..s400 (replicate-major, so a batch mixes sizes and configs)."""
    return (rep * len(SIZES) + SIZES.index(n)) * len(CONFIGS) + list(CONFIGS).index(config) + 1


def result_dir(config, n, rep):
    return RESULTS / config / f"n{n}_r{rep:02d}"


def render_settings(workdir, name):
    text = SETTINGS_TEMPLATE.read_text().replace("__TASK__", str(Path(workdir).resolve()))
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    path = SETTINGS_DIR / f"{name}.json"
    path.write_text(text)
    return path


def read_task_map():
    return pd.read_csv(TASK_MAP) if TASK_MAP.exists() else pd.DataFrame(
        columns=["task", "config", "n", "rep", "workdir", "attempt"])
