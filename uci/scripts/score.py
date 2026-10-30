"""Score every archived program: contract check, density test NLL on its fold (the score stated in
TASK.md; the one of the published competitors), flags, optimism gap, cost. The precision-aware NLL
(scripts/precision_score.py) is recorded as a secondary diagnostic only.

  python scripts/score.py [--jobs 16]   ->  results/per_fold.csv

Reads results/<config>/<dataset>_f<k>/{model.py, notes.md, run_summary.json} and data/uci/.
Deterministic: the same archived programs always give the same numbers.
"""
import argparse
import ast
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONFIGS, DATASETS, MAX_LITERALS, RESULTS, precision, split  # noqa: E402
from adr import contract  # noqa: E402
from precision_score import precision_nll  # noqa: E402

TRAIN_SAMPLE = 2000


def complexity(src):
    tree = ast.parse(src)
    return contract.count_literals(tree), len(list(ast.walk(tree)))


def nll(model, X, y):
    lp = np.array([float(np.asarray(model(x).logpdf(np.array([yi])))[0]) for x, yi in zip(X, y)])
    return -lp.mean(), int(np.sum(~np.isfinite(lp)))


def shape_flags(model, X, jitter, n_rows=20, n_grid=4000):
    """Median number of modes, and the largest fraction of rows sharing a narrow peak."""
    rng = np.random.default_rng(0)
    rows = X[rng.choice(len(X), size=min(n_rows, len(X)), replace=False)]
    modes, narrow = [], []
    for x in rows:
        d = model(x)
        lo, hi = np.quantile(np.asarray(d.sample(rng, 4000)), [0.005, 0.995])
        grid = np.linspace(lo - 5 * jitter, hi + 5 * jitter, n_grid)
        f = np.exp(np.asarray(d.logpdf(grid), dtype=float))
        peaks = 1 + np.flatnonzero((f[1:-1] > f[:-2]) & (f[1:-1] > f[2:]))
        modes.append(len(peaks))
        for k in peaks:
            below = f < f[k] / 2
            left, right = below[:k][::-1].argmax(), below[k:].argmax()
            if below[:k].any() and below[k:].any() and \
                    grid[k + right] - grid[k - 1 - left] < 7 * 2.355 * jitter:
                narrow.append((len(modes), grid[k]))
    n_modes = float(np.median(modes))
    locs = np.array([l for _, l in narrow])
    shared = max((len({r for r, l in narrow if abs(l - c) < 2 * jitter}) for c in locs),
                 default=0) / len(rows)
    flag = "lattice" if n_modes > 10 else "spike" if shared >= 0.5 else ""
    return n_modes, shared, flag


def evaluate(args):
    config, name, fold = args
    d = RESULTS / config / f"{name}_f{fold}"
    row = {"config": config, "dataset": name, "fold": fold}
    summ = d / "run_summary.json"
    if summ.exists():
        s = json.loads(summ.read_text())
        u = s.get("usage", {})
        row.update(minutes=s.get("minutes"), turns=s.get("num_turns"), attempts=s.get("attempts"),
                   run_status=s.get("status"), output_tokens=u.get("output_tokens"),
                   input_tokens=sum(u.get(k, 0) for k in ("input_tokens", "cache_read_input_tokens",
                                                          "cache_creation_input_tokens")),
                   cost_usd=s.get("cost_usd"))
    path = d / "model.py"
    if not path.exists():
        return {**row, "status": "missing"}
    (Xtr, ytr), (Xte, yte) = split(name, fold)
    try:
        row["n_literals"], row["ast_nodes"] = complexity(path.read_text())
    except SyntaxError as e:
        return {**row, "status": f"rejected: syntax error {e}"}
    probs = contract.check(path, Xtr.shape[1], 1, MAX_LITERALS, xs=Xtr)
    if probs:
        return {**row, "status": "rejected: " + "; ".join(probs)}
    model = contract.load_model(path)
    sub = np.random.default_rng(0).permutation(len(ytr))[:TRAIN_SAMPLE]
    try:  # a program that raises on a row scores nothing (operator decision, audit_notes.md)
        row["train_nll"], _ = nll(model, Xtr[sub], ytr[sub])
        row["test_nll"], row["n_nonfinite"] = nll(model, Xte, yte)  # density NLL (primary score)
        row["w"] = precision(name)
        row["test_nll_precision"] = precision_nll(model, Xte, yte, row["w"])  # secondary diagnostic
        row["n_modes"], row["shared_peak"], row["flag"] = shape_flags(model, Xte, 0.01 * ytr.std())
    except Exception as e:
        return {**row, "status": f"failed on data: {type(e).__name__}: {e}"}
    notes = d / "notes.md"
    m = re.search(r"expected_nll:\s*(-?[\d.eE+-]+)", notes.read_text()) if notes.exists() else None
    try:
        row["expected_nll"] = float(m.group(1)) if m else np.nan
    except ValueError:
        row["expected_nll"] = np.nan
    row["optimism_gap"] = row["test_nll"] - row["expected_nll"]
    return {**row, "status": "ok"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    jobs = [(c, n, k) for c in CONFIGS for n in DATASETS for k in range(1, 6)
            if (RESULTS / c / f"{n}_f{k}").exists()]
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(evaluate, jobs, chunksize=1))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "per_fold.csv", index=False)
    print(f"scored {len(df)} runs -> results/per_fold.csv")
    if len(df):
        print(df.groupby("config").status.value_counts().to_string())


if __name__ == "__main__":
    main()
