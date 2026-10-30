"""Shared definitions: paths, datasets, agent configurations, folds, task layout.

PKG  = this package (the folder containing README.md).
WS   = the neutral workspace seen by the agents (default ~/ws; override with ADR_WS).
       WS/env  Python environment used by agents and scripts
       WS/runs live task directories, one per (config, dataset, fold): WS/runs/<config>/tNNNN
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
DATA = PKG / "data" / "uci"
SNAPSHOT = PKG / "adr_snapshot"
RESULTS = PKG / "results"
SETTINGS_TEMPLATE = WS / "settings.template.json"  # installed by setup.sh
SETTINGS_DIR = WS / "settings"                     # one rendered file per task, outside task dirs
TMP_DIR = WS / "tmp"                               # one private harness temp dir per task
TASK_MAP = RESULTS / "task_map.csv"

# The 12 datasets of Table 3 whose target is continuous. SkillCraft and studentmath, whose targets
# are discrete (Table 3 of Franco & Tedesco, arXiv:2511.14455, which excludes them for the same
# reason), are not analysed: a density NLL is not defined for a discrete target. Their files stay
# in data/ unchanged (checksums) but are never used.
DATASETS = ["energy", "synchronous", "localization", "toxicity", "concrete", "slump",
            "forestfires", "NavalPropolsion", "sml2010", "Thermography", "support2",
            "superconductivity"]
CONFIGS = {  # name: (model id, budget in minutes)
    "opus-60": ("claude-opus-5-5", 60),
}
# Task numbering: t0001..t0120 are fold-major with two slots per (dataset, fold); opus-60 has the
# even numbers (t0002, t0004, ...). Kept fixed so that task ids match the archived runs.
SLOTS = ["sonnet-60", "opus-60"]
PRECISION = PKG / "data" / "uci_precision.csv"  # diagnostic only (lattice flags, secondary precision-aware column)
N_FOLDS = 5
MAX_LITERALS = 80

sys.path.insert(0, str(PKG))  # `adr` package (identical to adr_snapshot)


def load(name):
    """Covariates and target of a dataset (last column = target), as released."""
    d = pd.read_csv(DATA / f"{name}.csv")
    return d.iloc[:, :-1].to_numpy(dtype=float), d.iloc[:, -1].to_numpy(dtype=float)


def domain(name):
    """(d, 2) array of [min, max] of each covariate over the full dataset (all folds), as in the
    histogram range of Yang and van Leeuwen (2024); given to the agent and enforced by the
    contract check."""
    X, _ = load(name)
    return np.column_stack([X.min(axis=0), X.max(axis=0)])


def test_index(n, fold):
    """Test rows of `fold` (1-based): rep(1:5, each = ceiling(n/5))[1:n] == fold (run_cde.R)."""
    groups = np.repeat(np.arange(1, N_FOLDS + 1), int(np.ceil(n / N_FOLDS)))[:n]
    return np.flatnonzero(groups == fold)


def split(name, fold):
    X, y = load(name)
    te = test_index(len(y), fold)
    tr = np.setdiff1d(np.arange(len(y)), te)
    return (X[tr], y[tr]), (X[te], y[te])


def task_number(config, name, fold):
    """Deterministic, neutral task number (fold-major); see SLOTS."""
    c = SLOTS.index(config)
    return (fold - 1) * len(DATASETS) * len(SLOTS) + DATASETS.index(name) * len(SLOTS) + c + 1


def precision(name):
    """Recording precision w of the target of `name` (fixed in data/uci_precision.csv)."""
    t = pd.read_csv(PRECISION, index_col="dataset")
    return float(t.loc[name, "w"])


def result_dir(config, name, fold):
    return RESULTS / config / f"{name}_f{fold}"


def render_settings(workdir, name):
    """Sandbox settings of one agent: the template with the absolute path of its task dir."""
    text = SETTINGS_TEMPLATE.read_text().replace("__TASK__", str(Path(workdir).resolve()))
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    path = SETTINGS_DIR / f"{name}.json"
    path.write_text(text)
    return path


def read_task_map():
    return pd.read_csv(TASK_MAP) if TASK_MAP.exists() else pd.DataFrame(
        columns=["task", "config", "dataset", "fold", "workdir", "attempt"])
