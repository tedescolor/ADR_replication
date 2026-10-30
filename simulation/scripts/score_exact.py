"""Score every archived program with exact quantiles in place of Monte Carlo ones.

  PY scripts/score_exact.py [--jobs 16]   ->  results/per_rep_exact.csv

Same criteria, grid and quadrature as scripts/score.py (101 covariates, quantile levels 0.01..0.99,
Simpson with dx = 0.01), but the true quantiles are obtained by inverting the analytic conditional
distribution function, and the fitted quantiles by inverting the distribution function of each
program, instead of the Monte Carlo reference array qs_true.npy and 1,000 draws per covariate.
Both inversions compare F(y) with q in log space (see _terms), so that quantiles between well
separated modes are determined. Deterministic: no random numbers enter the scores.
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CONFIGS, QGRID, QUANTILE_LEVELS, REPLICATES, RESULTS, SIZES, XGRID, aqe,  # noqa: E402
                    awd, result_dir)
from adr import contract, dist  # noqa: E402


def _terms(d, y):
    """F(y) of a univariate Distribution, for a vector y, as c + sum(exp(pos)) - sum(exp(neg)):
    each component contributes its log cdf below its median and 1 minus its log survival function
    above it. Between well separated modes F(y) - q is far below double precision when computed
    directly (e.g. 0.5 + 1e-50), so the plain cdf leaves the median undetermined over the gap."""
    if isinstance(d, dist.Mixture):
        c, pos, neg = 0.0, [], []
        for k, comp in enumerate(d.components):
            w = float(np.ravel(d.w[k])[0])
            ck, pk, nk = _terms(comp, y)
            c = c + w * ck
            with np.errstate(divide="ignore"):  # a zero weight contributes nothing
                lw = np.log(w)
            pos += [lw + v for v in pk]
            neg += [lw + v for v in nk]
        return c, pos, neg
    f = d.d  # frozen scipy distribution (Normal, StudentT, SkewNormal)
    low = np.asarray(f.cdf(y), dtype=float) <= 0.5
    return (np.where(low, 0.0, 1.0), [np.where(low, f.logcdf(y), -np.inf)],
            [np.where(low, -np.inf, f.logsf(y))])


def _below(d, y, q):
    """Whether F(y) < q, elementwise, decided in log space."""
    c, pos, neg = _terms(d, y)
    c = c - q
    with np.errstate(divide="ignore", invalid="ignore"):
        lc = np.log(np.abs(c))
    lp = np.logaddexp.reduce(np.stack(pos + [np.where(c > 0, lc, -np.inf)]), axis=0)
    ln = np.logaddexp.reduce(np.stack(neg + [np.where(c < 0, lc, -np.inf)]), axis=0)
    return lp < ln


def quantiles(d, levels, iters=100):
    """inf{y : F(y) >= q} for each level q, by bisection with the log-space comparison."""
    q = np.asarray(levels, dtype=float)
    s = np.asarray(d.sample(np.random.default_rng(0), 2000), dtype=float).ravel()
    span = s.max() - s.min() + 1.0
    lo, hi = np.full_like(q, s.min() - 10 * span), np.full_like(q, s.max() + 10 * span)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        below = _below(d, mid, q)
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def true_law(x):
    """The branched law (common.f): normal for x < 0.5, an equal mixture of two normals with
    common scale for x >= 0.5."""
    m1 = 10 * x * (x - 0.5) * (1.5 - x)
    m2 = 10 * x * (x - 0.5) * (0.8 - x)
    s = 0.3 * (1.3 - x)
    if x < 0.5:
        return dist.Normal(m1, s)
    return dist.Mixture([0.5, 0.5], [dist.Normal(m1, s), dist.Normal(m2, s)])


def true_quantiles():
    """Exact quantiles of the branched law on the 101 x 99 grid. For x >= 0.5 the law is symmetric
    about (m1 + m2) / 2, which is therefore its median; the check below confirms it."""
    Q = np.stack([quantiles(true_law(x), QGRID) for x in XGRID])
    m1 = 10 * XGRID * (XGRID - 0.5) * (1.5 - XGRID)
    m2 = 10 * XGRID * (XGRID - 0.5) * (0.8 - XGRID)
    median = np.where(XGRID < 0.5, m1, (m1 + m2) / 2)
    assert np.abs(Q[:, list(QGRID).index(0.5)] - median).max() < 1e-9
    xs = XGRID < 0.5
    assert np.abs(Q[xs] - norm.ppf(QGRID[None, :], m1[xs, None], 0.3 * (1.3 - XGRID[xs, None]))).max() < 1e-9
    return Q


Q_EXACT = true_quantiles()


def evaluate(job):
    config, n, rep = job
    model = contract.load_model(result_dir(config, n, rep) / "model.py")
    pred = np.stack([quantiles(model(np.array([x])), QGRID) for x in XGRID])
    err = np.abs(Q_EXACT - pred)
    row = {"config": config, "n": n, "rep": rep, "AWD": awd(err)}
    for q in QUANTILE_LEVELS:
        row[f"AQE_{q:.2f}"] = aqe(err, q)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    jobs = [(c, n, r) for c in CONFIGS for n in SIZES for r in range(REPLICATES)
            if (result_dir(c, n, r) / "model.py").exists()]
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(evaluate, jobs, chunksize=4))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "per_rep_exact.csv", index=False)
    print(f"scored {len(df)} runs -> results/per_rep_exact.csv")
    print(df.groupby(["config", "n"])[["AWD"] + [f"AQE_{q:.2f}" for q in QUANTILE_LEVELS]].mean().round(4))


if __name__ == "__main__":
    main()
