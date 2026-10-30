"""Validate the metric, the data files and the true density before any agent runs (README step 3).

  PY scripts/validate.py   ->  results/validation.json, prints PASS/FAIL

Checks:
  1. the authors' metric, applied to their own KCDE samples (data/blobs_ckde.npy), gives the KCDE AWD
     of Table 2 (0.684): the test file, the sample layout and our W1 agree with the notebook;
  2. the true density (ported from the authors' R code) integrates to 1 on a grid at several x, and its
     mean log-density on the authors' true test samples matches that on fresh samples from the port;
  3. the floor of the metric: 1000 true samples against the authors' 1000 true samples gives the AWD
     of a perfect estimator (not 0), also at the three covariates of Table 2;
  4. a deliberately wrong program (one bivariate normal with the marginal moments) passes the contract
     and scores much worse than the floor, with a positive excess NLL.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DATA, RESULTS, THREE_X, avg_w1, per_x_w1, sample_program, test_set,  # noqa: E402
                    three_x_set, train_set, true_logpdf, true_sample)
from adr import contract  # noqa: E402

WRONG_PROGRAM = '''from adr.dist import MultivariateNormal


def model(x):
    return MultivariateNormal([0.0, 0.0], [[3.3, 0.0], [0.0, 3.4]])
'''


def main():
    out = {}
    Xte, Yte = test_set()
    X3, Y3 = three_x_set()
    # 1. the metric on the authors' KCDE samples
    ckde = np.load(DATA / "blobs_ckde.npy")[:, :, :2]
    out["kcde_awd"] = avg_w1(Yte, ckde)
    # 2. true density
    xs = np.vstack([Xte[:6], X3])
    g = np.linspace(-9, 9, 720)  # even number of points: the grid avoids the origin, where the ring density is singular
    G = np.array(np.meshgrid(g, g)).reshape(2, -1).T
    out["density_integrals"] = [float(np.exp(true_logpdf(G, x)).sum() * (g[1] - g[0]) ** 2) for x in xs]
    rng = np.random.default_rng(0)
    ref = np.mean([true_logpdf(Yte[i], Xte[i]).mean() for i in range(len(Xte))])
    fresh = np.mean([true_logpdf(true_sample(Xte[i], 1000, rng), Xte[i]).mean() for i in range(len(Xte))])
    out["mean_logpdf_authors_samples"], out["mean_logpdf_fresh_samples"] = float(ref), float(fresh)
    # 3. floor of the metric (1000 vs 1000 true samples)
    fl = np.stack([true_sample(x, 1000, rng) for x in Xte])
    fl3 = np.stack([true_sample(x, 1000, rng) for x in X3])
    out["floor"] = {"AWD": avg_w1(Yte, fl), **dict(zip([f"W1_{k}" for k in THREE_X], map(float, per_x_w1(Y3, fl3))))}
    # 4. a wrong program through the same path as the scorer
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "model.py"
        p.write_text(WRONG_PROGRAM)
        Xtr, _ = train_set()
        probs = contract.check(p, 5, 2, 80, xs=Xtr)
        model = contract.load_model(p)
        wr = {"contract_ok": not probs,
              "AWD": avg_w1(Yte, sample_program(model, Xte, np.random.default_rng(1))),
              "excess_nll": float(np.mean([true_logpdf(y, x) - np.asarray(model(x).logpdf(y)).reshape(-1)
                                           for x, y in zip(Xte, Yte)]))}
    out["wrong_model"] = wr
    checks = {
        "metric_reproduces_kcde_0.684": abs(out["kcde_awd"] - 0.684) < 1e-3,
        "true_density_integrates_to_1": all(abs(v - 1) < 5e-3 for v in out["density_integrals"]),
        "true_density_matches_authors_samples": abs(ref - fresh) < 0.03,
        "floor_below_0.25": out["floor"]["AWD"] < 0.25,
        "wrong_model_contract_ok": wr["contract_ok"],
        "wrong_model_much_worse": wr["AWD"] > 3 * out["floor"]["AWD"] and wr["excess_nll"] > 0.3,
    }
    out["checks"], out["passed"] = checks, all(checks.values())
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "validation.json").write_text(json.dumps(out, indent=1, default=float))
    for k, v in checks.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    print(f"KCDE AWD recomputed: {out['kcde_awd']:.4f} (Table 2: 0.684)")
    print(f"density integrals: {np.round(out['density_integrals'], 4)}")
    print(f"floor (true model, 1000 vs 1000): AWD {out['floor']['AWD']:.3f}; per-x W1 {np.round(list(out['floor'].values())[1:], 3)}")
    print(f"wrong model: AWD {wr['AWD']:.3f}, excess NLL {wr['excess_nll']:.3f}")
    print("ALL CHECKS PASSED" if out["passed"] else "VALIDATION FAILED")
    sys.exit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
