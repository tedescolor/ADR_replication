"""Dequantize the two count responses of data/dataset.csv, once, with a fixed seed.

  python3 scripts/make_dequantized.py            # writes data/dataset_dq.csv
  python3 scripts/make_dequantized.py --check    # verifies the file on disk

The responses of dataset.csv are y = log(1 + c) of nonnegative integer counts c (30-day page
views and 12-month edits), so their conditional law is discrete and a density NLL is not defined
(a density can put arbitrarily high spikes on the atoms). Each response is replaced by
    y_dq = log(1 + c + U),   U ~ Uniform(0, 1), independent across rows and responses,
which has a density: on [log(1 + c), log(2 + c)) it equals P(C = c) exp(y_dq), so the density
NLL of y_dq is the NLL of the counts minus a term that does not depend on the estimator.
The counts are recovered from the four-decimal values as c = round(expm1(y)); every value
round-trips exactly. Standard library only (random.Random is stable across Python versions).
Covariates are copied unchanged. Rows keep their order, so the folds are unchanged.
"""
import argparse
import csv
import math
import random
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
SEED = 20261023
RESPONSES = ["y1", "y2"]


def build():
    with open(DATA / "dataset.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    rng = random.Random(SEED)
    out = []
    for r in rows:
        new = dict(r)
        for col in RESPONSES:
            y = float(r[col])
            c = round(math.expm1(y))
            assert c >= 0 and abs(round(math.log1p(c), 4) - y) < 1e-9, (col, y)
            new[col] = f"{math.log1p(c + rng.random()):.8f}"
        out.append(new)
    return list(rows[0].keys()), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    cols, out = build()
    path = DATA / "dataset_dq.csv"
    if a.check:
        with open(path, newline="") as f:
            disk = list(csv.DictReader(f))
        assert disk == out, "data/dataset_dq.csv does not match make_dequantized.py"
        print(f"{path.name}: OK ({len(out)} rows)")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    print(f"wrote {path} ({len(out)} rows, seed {SEED})")


if __name__ == "__main__":
    main()
