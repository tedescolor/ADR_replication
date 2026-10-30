"""Score every archived program with the metric of Table 2 of the CPFN paper (and ours).

  PY scripts/score.py [--jobs 4]   ->  results/per_run.csv, results/per_x.csv

For results/<config>/run<k>/model.py: contract check (bivariate response); AWD exactly as
blobs.ipynb (1000 samples from the program at each of the 30 fixed test covariates, mean over x of
the 1-Wasserstein distance to the 1000 true samples, POT emd2, Euclidean cost); W1 at the three
covariates of Table 2 (x_ring, x_trans, x_blobs; true samples regenerated with the authors' R code);
excess NLL of the program over the true density on the 30 x 1000 true test samples (ours, not in the
paper); number of constants; optimism gap (test NLL minus the agent's expected_nll); cost.
Deterministic: the sampling seed is fixed (SAMPLING_SEED).
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
from common import (CONFIGS, MAX_LITERALS, REPS, RESULTS, THREE_X, avg_w1, per_x_w1,  # noqa: E402
                    result_dir, sample_program, test_set, three_x_set, train_set, true_logpdf)
from adr import contract  # noqa: E402

SAMPLING_SEED = 123


def evaluate(job):
    config, rep = job
    d = result_dir(config, rep)
    row = {"config": config, "rep": rep}
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
        return {**row, "status": "missing"}, []
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as e:
        return {**row, "status": f"rejected: syntax error {e}"}, []
    row["n_literals"], row["ast_nodes"] = contract.count_literals(tree), len(list(ast.walk(tree)))
    Xtr, _ = train_set()
    probs = contract.check(path, Xtr.shape[1], 2, MAX_LITERALS, xs=Xtr)
    if probs:
        return {**row, "status": "rejected: " + "; ".join(probs)}, []
    model = contract.load_model(path)
    Xte, Yte = test_set()
    X3, Y3 = three_x_set()
    rng = np.random.default_rng(SAMPLING_SEED)
    try:
        Ypred = sample_program(model, Xte, rng)
        Y3pred = sample_program(model, X3, rng)
        w1 = per_x_w1(Yte, Ypred)
        row["AWD"] = float(w1.mean())
        for name, v in zip(THREE_X, per_x_w1(Y3, Y3pred)):
            row[f"W1_{name}"] = float(v)
        lp = np.stack([np.asarray(model(x).logpdf(y), dtype=float).reshape(-1) for x, y in zip(Xte, Yte)])
        lt = np.stack([true_logpdf(y, x) for x, y in zip(Xte, Yte)])
    except Exception as e:
        return {**row, "status": f"failed on data: {type(e).__name__}: {e}"}, []
    row["test_nll"] = float(-lp.mean())
    row["excess_nll"] = float(np.mean(lt - lp))
    row["n_nonfinite"] = int(np.sum(~np.isfinite(lp)))
    notes = d / "notes.md"
    m = re.search(r"expected_nll:\s*(-?[\d.eE+-]+)", notes.read_text()) if notes.exists() else None
    try:
        row["expected_nll"] = float(m.group(1)) if m else np.nan
    except ValueError:
        row["expected_nll"] = np.nan
    row["optimism_gap"] = row["test_nll"] - row["expected_nll"]
    px = [{"config": config, "rep": rep, "test_index": i, "W1": float(w)} for i, w in enumerate(w1)]
    return {**row, "status": "ok"}, px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    jobs = [(c, r) for c in CONFIGS for r in range(REPS) if result_dir(c, r).exists()]
    with ProcessPoolExecutor(a.jobs) as ex:
        out = list(ex.map(evaluate, jobs, chunksize=1))
    df = pd.DataFrame([o[0] for o in out])
    df.to_csv(RESULTS / "per_run.csv", index=False)
    pd.DataFrame([r for o in out for r in o[1]]).to_csv(RESULTS / "per_x.csv", index=False)
    print(f"scored {len(df)} runs -> results/per_run.csv, results/per_x.csv")
    if len(df):
        print(df.groupby("config").status.value_counts().to_string())


if __name__ == "__main__":
    main()
