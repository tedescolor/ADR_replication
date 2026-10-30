"""CPFN baseline on the five folds of data/dataset_dq.csv (Python package cpfn 1.0.5, CPU).

  python scripts/baselines/cpfn_fit.py [--folds 1 2 3 4 5] [--m 500] [--device cuda] [--data ...] [--out ...]

Settings of the UCI study of Franco & Tedesco (Section 5 of arXiv:2511.14455): three hidden gelu
layers of width 50 for both submodules, terminal activation for psi_U, rank r = 50,
U ~ N(0, I_q), R latent draws per observation in the loss (--m; the paper uses 100, the
reported run 500), initial bandwidth 0.05 (learned),
delta = 1e-15, Adam with the default learning rate for 2000 epochs, 10% of the training rows held
out to keep the best state. Inputs and outputs are z-scored inside the package (training rows
only); the package returns the log-density of the standardized response, so the density on the
original scale subtracts sum_j log sd(y_j) (change of variables, as in the CPFN paper).
The test density is the kernel mixture of the fitted CPFN evaluated with S = 20000 latent draws
per test row (the paper does not state S; 20000 makes the Monte Carlo error negligible).
Writes <out>/cpfn_m<R>_per_row.csv (fold, row, logpdf) and <out>/cpfn_m<R>_per_fold.csv.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from cpfn import CPFN

X_COLS = ["x1", "x2", "x3", "x4", "x5", "x6"]
Y_COLS = ["y1", "y2"]
S_EVAL = 20000
SEED = 20261023


def folds(n, k=5):
    return np.repeat(np.arange(1, k + 1), int(np.ceil(n / k)))[:n]


def test_logpdf(model, X, Y, device, s=S_EVAL, chunk=2000):
    """log p(y | x) on the original scale, by averaging the Gaussian kernel over s latent draws."""
    out = []
    with torch.no_grad():
        for x, y in zip(X, Y):
            xt = torch.tensor(x[None, :], dtype=torch.float32, device=device)
            yt = torch.tensor(y[None, :], dtype=torch.float32, device=device)
            parts = []
            for _ in range(s // chunk):  # log-mean-exp over chunks of draws
                parts.append(model.logdensity(xt, yt, m=chunk).reshape(-1)[0].item())
            lp = np.logaddexp.reduce(parts) - np.log(len(parts))
            out.append(lp - float(torch.log(model.y_std).sum()))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--data", default="data/dataset_dq.csv")
    ap.add_argument("--out", default="results/baselines")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--m", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    torch.set_num_threads(2)
    d = pd.read_csv(a.data)
    X, Y = d[X_COLS].to_numpy(np.float32), d[Y_COLS].to_numpy(np.float32)
    f = folds(len(d))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, summ = [], []
    for k in a.folds:
        torch.manual_seed(SEED + k)
        torch.cuda.manual_seed_all(SEED + k)
        np.random.seed(SEED + k)
        tr, te = f != k, f == k
        t0 = time.time()
        model = CPFN(d=X.shape[1], q=Y.shape[1], r=50, width=50, hidden_layers=3,
                     latent_dist="normal", learn_eps=True, eps_init=0.05, delta=1e-15,
                     psi_final_activation=True)
        model.to(a.device)
        model.fit(X[tr], Y[tr], epochs=a.epochs, lr=1e-3, m=a.m, h0=0.05, val_split=0.1)
        model.freeze()
        lp = test_logpdf(model, X[te], Y[te], a.device)
        idx = np.flatnonzero(te)
        rows += [{"fold": k, "row": int(i), "logpdf": float(v)} for i, v in zip(idx, lp)]
        summ.append({"fold": k, "nll": float(-lp.mean()), "minutes": round((time.time() - t0) / 60, 2),
                     "m": a.m, "device": a.device,
                     "bandwidth": ";".join(f"{e:.4g}" for e in model.eps().cpu().numpy())})
        print(f"fold {k}: test NLL {-lp.mean():.4f}  ({summ[-1]['minutes']} min)", flush=True)
    pd.DataFrame(rows).to_csv(out / f"cpfn_m{a.m}_per_row.csv", index=False)
    pd.DataFrame(summ).to_csv(out / f"cpfn_m{a.m}_per_fold.csv", index=False)


if __name__ == "__main__":
    main()
