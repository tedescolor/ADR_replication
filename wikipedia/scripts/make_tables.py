"""Comparison table: test NLL per fold and mean (sd over folds) for KCDE, CPFN and ADR.

  PY scripts/make_tables.py   # reads results/per_fold.csv and results/baselines/*; writes results/tables/

CPFN is the run with R = 500 latent draws (cpfn_m500_per_fold.csv); the run with the paper's
R = 100 is reported as a diagnostic column. A fold on which an ADR program returns a non-finite
parameter is not scored (as in the UCI study): its cell is empty, the NLL over its finite test rows
is given in a diagnostic column, and the summary rows give the mean over all scored folds and over
the folds scored for every method.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONFIGS, RESULTS  # noqa: E402

LABEL = {"opus-60": "ADR: Opus 5.5"}


def main():
    b = RESULTS / "baselines"
    cols = {}
    kc = pd.concat([pd.read_csv(p) for p in sorted(b.glob("kcde_fold[0-9].csv"))])
    cols["KCDE"] = kc.set_index("fold").nll
    cols["CPFN"] = pd.read_csv(b / "cpfn_m500_per_fold.csv").set_index("fold").nll
    pf = pd.read_csv(RESULTS / "per_fold.csv")
    for c in CONFIGS:
        r = pf[pf.config == c].set_index("fold")
        cols[LABEL[c]] = r.test_nll.where(r.status == "ok")
    main_cols = list(cols)
    cols["CPFN (R = 100)"] = pd.read_csv(b / "cpfn_m100_per_fold.csv").set_index("fold").nll
    for c in CONFIGS:
        r = pf[pf.config == c].set_index("fold")
        cols[LABEL[c] + " (finite rows)"] = r.test_nll_finite_rows
    t = pd.DataFrame(cols).sort_index()
    common = t[main_cols].dropna().index
    summ = pd.DataFrame({
        "mean (scored folds)": t.mean(),
        "sd (scored folds)": t.std(ddof=1),
        f"mean (folds {','.join(map(str, common))})": t.loc[common].mean(),
    }).T
    t.index = [f"fold {k}" for k in t.index]
    out = RESULTS / "tables"
    out.mkdir(exist_ok=True)
    full = pd.concat([t, summ])
    full.round(4).to_csv(out / "table_nll.csv")
    print(full.round(3).to_string())


if __name__ == "__main__":
    main()
