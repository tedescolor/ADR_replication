"""Score every archived program with the CPFN paper's metrics (and ours).

  PY scripts/score.py [--jobs 16]   ->  results/per_rep.csv

For results/<config>/n<n>_r<rep>/model.py: contract check; AWD and AQE(0.1, 0.25, 0.5, 0.75, 0.9)
exactly as in cpfn/branched.ipynb (1000 samples from the program at each of the 101 grid points,
empirical quantiles at the 99 levels, Simpson with dx = 0.01, true quantiles from qs_true.npy);
test NLL and excess NLL over the true density on 10,000 independent rows; number of constants;
optimism gap (test NLL minus the agent's expected_nll); cost from run_summary.json.
Deterministic: the sampling seed of each program is fixed (see SAMPLING_SEED).
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
from common import (CONFIGS, MAX_LITERALS, QUANTILE_LEVELS, REPLICATES, RESULTS, SIZES, aqe,  # noqa: E402
                    awd, quantile_errors, result_dir, test_set, training_seed, training_set,
                    true_logpdf)
from adr import contract  # noqa: E402


def SAMPLING_SEED(n, rep):
    return 10**6 + training_seed(n, rep)


def evaluate(job):
    config, n, rep = job
    d = result_dir(config, n, rep)
    row = {"config": config, "n": n, "rep": rep}
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
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as e:
        return {**row, "status": f"rejected: syntax error {e}"}
    row["n_literals"], row["ast_nodes"] = contract.count_literals(tree), len(list(ast.walk(tree)))
    xs, ys = training_set(n, rep)
    probs = contract.check(path, 1, 1, MAX_LITERALS, xs=xs.reshape(-1, 1))
    if probs:
        return {**row, "status": "rejected: " + "; ".join(probs)}
    model = contract.load_model(path)
    err = quantile_errors(model, np.random.default_rng(SAMPLING_SEED(n, rep)))
    row["AWD"] = awd(err)
    for q in QUANTILE_LEVELS:
        row[f"AQE_{q:.2f}"] = aqe(err, q)
    tx, ty = test_set(n, rep)
    lp = np.array([float(np.asarray(model(np.array([x])).logpdf(np.array([y])))[0])
                   for x, y in zip(tx, ty)])
    row["test_nll"] = -lp.mean()
    row["excess_nll"] = float(np.mean(true_logpdf(ty, tx) - lp))
    row["n_nonfinite"] = int(np.sum(~np.isfinite(lp)))
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
    jobs = [(c, n, r) for c in CONFIGS for n in SIZES for r in range(REPLICATES)
            if result_dir(c, n, r).exists()]
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(evaluate, jobs, chunksize=1))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "per_rep.csv", index=False)
    print(f"scored {len(df)} runs -> results/per_rep.csv")
    if len(df):
        print(df.groupby("config").status.value_counts().to_string())


if __name__ == "__main__":
    main()
