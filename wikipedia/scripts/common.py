"""Shared definitions of the new-dataset study: paths, configuration, folds, data, task names.

PKG  = this package. WS = neutral workspace seen by the agents (default ~/ws; ADR_WS overrides):
       WS/env (Python), WS/runs/<config>/tNN (task dirs), WS/settings, WS/tmp, WS/done (finished
       task dirs, locked).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1]
WS = Path(os.environ.get("ADR_WS", "~/ws")).expanduser()
ENV_PY = WS / "env" / "bin" / "python"
RUNS = WS / "runs"
DONE = WS / "done"
SNAPSHOT = PKG / "adr_snapshot"
RESULTS = PKG / "results"
DATA = PKG / "data"
SETTINGS_TEMPLATE = WS / "settings.template.json"
SETTINGS_DIR = WS / "settings"
TMP_DIR = WS / "tmp"
TASK_MAP = RESULTS / "task_map.csv"

CONFIGS = {  # name: (model id, budget in minutes), as in the UCI study
    "opus-60": ("claude-opus-5-5", 60),
}
MAX_LITERALS = 80
N = 1000
FOLDS = 5
X_COLS = ["x1", "x2", "x3", "x4", "x5", "x6"]
Y_COLS = ["y1", "y2"]

sys.path.insert(0, str(PKG))  # `adr` package (identical to adr_snapshot)


def data():
    """The dequantized dataset (data/dataset_dq.csv): X (1000, 6), Y (1000, 2), in file order."""
    d = pd.read_csv(DATA / "dataset_dq.csv")
    assert len(d) == N and list(d.columns) == X_COLS + Y_COLS
    return d[X_COLS].to_numpy(float), d[Y_COLS].to_numpy(float)


def domain():
    """(6, 2) array of [min, max] of each covariate over the full dataset (all folds)."""
    X, _ = data()
    return np.column_stack([X.min(axis=0), X.max(axis=0)])


def fold_of_rows(n=N, k=FOLDS):
    """Fold label of every row: R's rep(1:k, each = ceiling(n/k))[1:n], contiguous blocks as in
    the UCI study (uci/scripts/common.py)."""
    return np.repeat(np.arange(1, k + 1), int(np.ceil(n / k)))[:n]


def split(fold):
    """(X_train, Y_train, X_test, Y_test) of fold 1..5."""
    X, Y = data()
    f = fold_of_rows()
    tr, te = f != fold, f == fold
    return X[tr], Y[tr], X[te], Y[te]


def task_id(config, fold):
    """Neutral task name: t01..t05 (one config)."""
    return f"t{fold:02d}"


def result_dir(config, fold):
    return RESULTS / config / f"fold{fold}"


def render_settings(workdir, name):
    text = SETTINGS_TEMPLATE.read_text().replace("__TASK__", str(Path(workdir).resolve()))
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    path = SETTINGS_DIR / f"{name}.json"
    path.write_text(text)
    return path


def read_task_map():
    return pd.read_csv(TASK_MAP) if TASK_MAP.exists() else pd.DataFrame(
        columns=["task", "config", "fold", "workdir", "attempt"])
