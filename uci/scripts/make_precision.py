"""Recording precision w of each target, fixed before any agent runs.

  python scripts/make_precision.py           # writes data/uci_precision.csv
  python scripts/make_precision.py --check   # verifies the shipped file (used by setup.sh)

For each dataset, from the noise-free files in data/uci_original/ (same rows as data/uci/):
  h = smallest gap between distinct target values (values rounded to 10 decimals),
  s = 0.01 sd(y), the sd of the noise the CDTree authors added (add_small_noise_to_data.R),
  w = max(h, s).
The released data cannot resolve anything finer than the added noise, so w is never below s;
on targets recorded on a coarse lattice (h >> s) w is the lattice spacing. The file also checks
that data/uci_original/ and data/uci/ differ by noise of exactly this size, row by row.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, DATASETS, PKG, PRECISION  # noqa: E402

ORIG = PKG / "data" / "uci_original"


def compute():
    rows = []
    for name in DATASETS:
        yo = pd.read_csv(ORIG / f"{name}.csv").iloc[:, -1].to_numpy(float)
        yj = pd.read_csv(DATA / f"{name}.csv").iloc[:, -1].to_numpy(float)
        s = 0.01 * yo.std(ddof=1)
        z = (yj - yo) / s
        if len(yo) != len(yj) or not (0.9 < z.std() < 1.1 and np.abs(z).max() < 6):
            raise SystemExit(f"{name}: data/uci_original does not match data/uci row by row")
        h = float(np.min(np.diff(np.unique(np.round(yo, 10)))))
        rows.append({"dataset": name, "h": h, "s": s, "w": max(h, s), "h_over_s": h / s,
                     "lattice": bool(h >= 5 * s)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    t = compute()
    if a.check:
        shipped = pd.read_csv(PRECISION)
        if not np.allclose(shipped.w.to_numpy(), t.w.to_numpy(), rtol=1e-9, atol=0):
            raise SystemExit("data/uci_precision.csv does not match data/uci_original/")
        print("OK: data/uci_precision.csv matches data/uci_original/")
        return
    t.to_csv(PRECISION, index=False, float_format="%.10g")
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
