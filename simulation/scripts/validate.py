"""Validate the ported data generation and metrics before any agent runs (README step 3).

  PY scripts/validate.py   ->  results/validation.json, prints PASS/FAIL

Checks:
  1. every AQE quantile level is exactly on the grid (the notebook's argmax(qgrid == q0));
  2. the true model, written as an ADR program, scores AWD and AQE at the Monte Carlo floor and an
     excess NLL of about 0 (its quantiles differ from qs_true.npy only by sampling noise);
  3. qs_true.npy agrees with quantiles recomputed from the generator (same noise level);
  4. a deliberately wrong program (a single Normal with the marginal mean and sd) scores much
     worse; and the training sets are deterministic (same seed -> identical rows).
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (QGRID, QUANTILE_LEVELS, Q_TRUE, RESULTS, XGRID, aqe, awd, q_index,  # noqa: E402
                    quantile_errors, test_set, training_set, true_logpdf, true_sample)
from adr import contract  # noqa: E402

TRUE_PROGRAM = '''from adr.dist import Normal, Mixture


def model(x):
    """The data-generating process of the CPFN univariate test case."""
    t = x[0]
    e1 = 10 * t * (t - 0.5) * (1.5 - t)
    e2 = 10 * t * (t - 0.5) * (0.8 - t)
    s = 0.3 * (1.3 - t)
    if t < 0.5:
        return Normal(e1, s)
    return Mixture([0.5, 0.5], [Normal(e1, s), Normal(e2, s)])
'''
WRONG_PROGRAM = '''from adr.dist import Normal


def model(x):
    return Normal(0.2, 0.6)
'''


def score_text(src):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "model.py"
        p.write_text(src)
        probs = contract.check(p, 1, 1, 80, xs=np.random.default_rng(0).random((20, 1)))
        model = contract.load_model(p)
        err = quantile_errors(model, np.random.default_rng(1))
        xs, ys = test_set(250, 0, size=5000)
        lp = np.array([float(model(np.array([x])).logpdf(np.array([y]))[0]) for x, y in zip(xs, ys)])
        return {"contract_ok": not probs, "AWD": awd(err),
                **{f"AQE_{q:.2f}": aqe(err, q) for q in QUANTILE_LEVELS},
                "excess_nll": float(np.mean(true_logpdf(ys, xs) - lp))}


def main():
    out = {}
    out["levels_on_grid"] = all(QGRID[q_index(q)] == q for q in QUANTILE_LEVELS)
    out["true_model"] = score_text(TRUE_PROGRAM)
    out["wrong_model"] = score_text(WRONG_PROGRAM)
    # qs_true.npy vs quantiles recomputed from the generator (1000 draws per x, as in the notebook)
    Q = np.load(Q_TRUE)
    np.random.seed(12345)
    Qr = np.array([np.quantile([true_sample(x) for _ in range(1000)], QGRID) for x in XGRID])
    out["qs_true_vs_regenerated_awd"] = awd(np.abs(Q - Qr))
    a, b = training_set(500, 3), training_set(500, 3)
    out["training_deterministic"] = bool(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))
    t, w = out["true_model"], out["wrong_model"]
    checks = {
        "levels_on_grid": out["levels_on_grid"],
        "true_model_contract_ok": t["contract_ok"],
        "true_model_awd_at_floor": t["AWD"] < 0.03,
        "true_model_excess_nll_zero": abs(t["excess_nll"]) < 1e-9,
        "qs_true_matches_generator": out["qs_true_vs_regenerated_awd"] < 0.03,
        "wrong_model_much_worse": w["AWD"] > 5 * t["AWD"] and w["excess_nll"] > 0.1,
        "training_deterministic": out["training_deterministic"],
    }
    out["checks"], out["passed"] = checks, all(checks.values())
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "validation.json").write_text(json.dumps(out, indent=1, default=float))
    for k, v in checks.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    print(f"true model: AWD {t['AWD']:.4f}, AQE(0.5) {t['AQE_0.50']:.4f}, excess NLL {t['excess_nll']:.2e}")
    print(f"wrong model: AWD {w['AWD']:.4f}, excess NLL {w['excess_nll']:.3f}")
    print(f"qs_true.npy vs regenerated quantiles: AWD {out['qs_true_vs_regenerated_awd']:.4f}")
    print("ALL CHECKS PASSED" if out["passed"] else "VALIDATION FAILED")
    sys.exit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
