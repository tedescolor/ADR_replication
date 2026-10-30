"""Shared definitions of the multivariate re-analysis: paths, configurations, tasks, the authors'
fixed training and test data, their metric (AWD, notebook blobs.ipynb) and the true density.

PKG  = this package. WS = neutral workspace seen by the agents (default ~/ws; ADR_WS overrides):
       WS/env (Python), WS/runs/<config>/sNNN (task dirs), WS/settings, WS/tmp.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm

PKG = Path(__file__).resolve().parents[1]
WS = Path(os.environ.get("ADR_WS", "~/ws")).expanduser()
ENV_PY = WS / "env" / "bin" / "python"
RUNS = WS / "runs"
SNAPSHOT = PKG / "adr_snapshot"
RESULTS = PKG / "results"
DATA = PKG / "data"
SETTINGS_TEMPLATE = WS / "settings.template.json"
SETTINGS_DIR = WS / "settings"
TMP_DIR = WS / "tmp"
TASK_MAP = RESULTS / "task_map.csv"

CONFIGS = {  # name: (model id, budget in minutes)
    "sonnet-60": ("claude-sonnet-5-5", 60),
    "opus-60": ("claude-opus-5-5", 60),
}
REPS = 5                 # agent runs per config on the authors' single dataset: rep 0 is the run of Table 2; reps 1-4 (Opus 5.5 only) measure agent variability
MAX_LITERALS = 80
N_TRAIN = 5000
M_SAMPLES = 1000         # samples per test x, as in blobs.ipynb (cpfn.sample(x, m = 1000))
THREE_X = {              # the three covariate vectors of Table 2 (Rcode/generate_multivariate_data.R)
    "x_ring": [2.0, 1.0, 0.5, -0.3, -1.0],
    "x_trans": [0.0, 0.0, 0.0, 0.0, 0.0],
    "x_blobs": [-2.0, -1.0, -0.5, 0.3, 1.0],
}

sys.path.insert(0, str(PKG))  # `adr` package (identical to adr_snapshot)


# ---------------------------------------------------------------- the authors' data
def train_set():
    """The 5000 training rows used by CPFN and KCDE: X (5000, 5), Y (5000, 2)."""
    d = pd.read_csv(DATA / "blobs_train.csv")
    assert len(d) == N_TRAIN
    return d[[f"X{i}" for i in range(1, 6)]].to_numpy(float), d[["Y1", "Y2"]].to_numpy(float)


def test_set():
    """The fixed test set: X (30, 5) and true samples Y (30, 1000, 2) (blobs_test.npy)."""
    a = np.load(DATA / "blobs_test.npy")
    return a[:, 0, 2:], a[:, :, :2]


def three_x_set():
    """True samples (3, 1000, 2) at THREE_X (the authors' 3-example file, set.seed(1234))."""
    a = pd.read_csv(DATA / "blobs_3ex_true.csv").to_numpy(float).reshape(3, 1000, 7)
    X = a[:, 0, 2:]
    assert np.allclose(X, np.array(list(THREE_X.values())))
    return X, a[:, :, :2]


# ---------------------------------------------------------------- the authors' metric (generative.py)
def W1(A, B):
    """1-Wasserstein distance between two point clouds (Euclidean cost, uniform weights, POT emd2)."""
    import ot
    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
    return ot.emd2(np.ones(len(A)) / len(A), np.ones(len(B)) / len(B), D)


def per_x_w1(y_true, y_pred):
    return np.array([W1(y_true[i], y_pred[i]) for i in range(len(y_true))])


def avg_w1(y_true, y_pred):
    """avgW1error of the authors: mean over x of W1(true samples, predicted samples)."""
    return float(per_x_w1(y_true, y_pred).mean())


def sample_program(model, X, rng, m=M_SAMPLES):
    """m samples from model(x) at every row of X: (len(X), m, 2)."""
    return np.stack([np.asarray(model(x).sample(rng, m), dtype=float).reshape(m, 2) for x in X])


# ---------------------------------------------------------------- the true density (generate_multivariate_data.R)
_BETA = np.array([1.2, -0.8, 0.6, -0.4, 0.9])
_GAMMA = np.array([-0.5, 1.1, -0.3, 0.7, 0.4])


def _sig(z):
    return 1 / (1 + np.exp(-z))


def _rot2(t):
    """The authors' rot2: matrix(c(cos, -sin, sin, cos), 2, 2) fills by column, i.e. [[c, s], [-s, c]]."""
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, s], [-s, c]])


def true_params(x):
    x = np.asarray(x, float)
    s1, s2 = _BETA @ x, _GAMMA @ x
    mu_sep = 2.0 + 1.2 * np.tanh(0.8 * s2)
    Q = _rot2(0.7 * s2)
    Sigma = Q @ np.diag([0.30**2, 0.10**2 * (1 + 0.8 * _sig(s2))]) @ Q.T
    m1 = _rot2(0.6 * s1) @ np.array([mu_sep, 0.0])
    return dict(w=_sig(s1), r=2.0 + 1.5 * np.tanh(0.7 * s1), sd=0.18 + 0.15 * _sig(1.2 * s2),
                m1=m1, m2=-m1, Sigma=Sigma)


def true_logpdf(y, x):
    """log f_{Y|X}(y | x) for y of shape (n, 2): ring (truncated-normal radius, uniform angle) + two blobs."""
    p = true_params(x)
    y = np.asarray(y, float).reshape(-1, 2)
    r = np.linalg.norm(y, axis=1)
    lg = norm.logpdf(r, p["r"], p["sd"]) - np.log1p(-norm.cdf(0.0, p["r"], p["sd"]))
    ring = lg - np.log(2 * np.pi * r)
    blobs = np.logaddexp(multivariate_normal.logpdf(y, p["m1"], p["Sigma"]),
                         multivariate_normal.logpdf(y, p["m2"], p["Sigma"])) + np.log(0.5)
    return np.logaddexp(np.log(p["w"]) + ring, np.log1p(-p["w"]) + blobs)


def true_sample(x, n, rng):
    p = true_params(x)
    ring = rng.random(n) < p["w"]
    R = rng.normal(p["r"], p["sd"], 4 * n)
    R = R[R >= 0][:n]
    A = rng.uniform(0, 2 * np.pi, n)
    ys = np.c_[R * np.cos(A), R * np.sin(A)]
    mu = np.where((rng.random(n) < 0.5)[:, None], p["m1"], p["m2"])
    return np.where(ring[:, None], ys, rng.multivariate_normal([0, 0], p["Sigma"], n) + mu)


# ---------------------------------------------------------------- tasks
def task_number(config, rep):
    """Neutral task number s001, s002, ... (run-major)."""
    return rep * len(CONFIGS) + list(CONFIGS).index(config) + 1


def result_dir(config, rep):
    return RESULTS / config / f"run{rep:02d}"


def render_settings(workdir, name):
    text = SETTINGS_TEMPLATE.read_text().replace("__TASK__", str(Path(workdir).resolve()))
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    path = SETTINGS_DIR / f"{name}.json"
    path.write_text(text)
    return path


def read_task_map():
    return pd.read_csv(TASK_MAP) if TASK_MAP.exists() else pd.DataFrame(
        columns=["task", "config", "rep", "workdir", "attempt"])
