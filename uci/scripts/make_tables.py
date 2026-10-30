"""Tables from results/per_fold.csv and published/table3_nll.csv.

  python scripts/make_tables.py   ->  results/tables/{table_uci.tex, table_uci.csv, table_uci_precision.csv,
                                       table_cost.csv, table_selfassess.csv, numbers.tex}

Agent columns use the density test NLL stated in TASK.md (test_nll), the score of the published
competitors; the precision-aware NLL of the same programs is in table_uci_precision.csv (secondary). Lattice datasets (data/uci_precision.csv) are
marked; ranks are given over all datasets and without the lattice datasets. CPFN (Franco & Tedesco,
Table 4; published/cpfn_uci.csv) is a published column like the others and is ranked.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONFIGS, DATASETS, PKG, PRECISION, RESULTS  # noqa: E402

INTERP = ["CADET", "CART-h", "CART-k", "CKDE", "LSCDE", "NKDE", "CDTree"]
BLACK = ["LinCDE", "MDN", "NF"]
AGENTS = list(CONFIGS)
SCORE = "test_nll"


def published():
    t = pd.read_csv(PKG / "published" / "table3_nll.csv", index_col="dataset")
    mean = t.apply(lambda c: c.str.extract(r"^(-?[\d.]+)")[0].astype(float))
    sd = t.apply(lambda c: c.str.extract(r"\((-?[\d.]+)\)")[0].astype(float))
    cpfn = pd.read_csv(PKG / "published" / "cpfn_uci.csv", index_col="dataset").CPFN
    mean["CPFN"], sd["CPFN"] = cpfn.reindex(mean.index), np.nan  # no standard deviations reported
    return mean, sd


def main():
    out = RESULTS / "tables"
    out.mkdir(parents=True, exist_ok=True)
    pf = pd.read_csv(RESULTS / "per_fold.csv")
    for col in ["minutes", "turns", "input_tokens", "output_tokens", "cost_usd", "n_literals",
                "optimism_gap", "flag", "test_nll", SCORE]:
        if col not in pf:
            pf[col] = np.nan
    ok = pf[pf.status == "ok"]
    mean, sd = published()
    idx = [d for d in DATASETS if d in mean.index]
    mean, sd = mean.loc[idx], sd.loc[idx]
    g = ok.groupby(["dataset", "config"])[SCORE]
    am, asd, an = g.mean().unstack(), g.std().unstack(), g.count().unstack()
    dens = ok.groupby(["dataset", "config"]).test_nll_precision.agg(["mean", "std"]).unstack()
    lattice = set(pd.read_csv(PRECISION).query("lattice").dataset)
    flags = ok[ok.flag.fillna("") != ""].groupby(["dataset", "config"]).size().unstack()
    nconst = ok.groupby(["dataset", "config"]).n_literals.median().unstack()
    # datasets without any scored run must still get a row (e.g. a partial or failed wave)
    flags = flags.reindex(index=DATASETS, columns=AGENTS)
    nconst = nconst.reindex(index=DATASETS, columns=AGENTS)
    for c in AGENTS:
        mean[c] = am[c] if c in am else np.nan
        sd[c] = asd[c] if c in asd else np.nan
    cols = INTERP + BLACK + ["CPFN"] + AGENTS
    complete = pd.DataFrame({c: (an[c] if c in an else 0) for c in AGENTS}).reindex(idx).fillna(0)
    full = mean[cols].copy()
    for c in AGENTS:  # an agent is ranked on a dataset only with all five folds scored
        full.loc[complete[c] < 5, c] = np.nan
    rank_all = full.rank(axis=1).mean()
    keep = [d for d in idx if d not in lattice]
    rank_nolat = full.loc[keep].rank(axis=1).mean()

    # CSV
    csv = pd.DataFrame({c: [("" if not np.isfinite(m) else f"{m:.3f}" + (f" ({v:.2f})" if np.isfinite(v) else ""))
                            for m, v in zip(mean[c], sd[c])] for c in cols}, index=mean.index)
    csv.index = [d + (" (lattice)" if d in lattice else "") for d in csv.index]
    csv.loc["rank (all)"] = rank_all.round(2)
    csv.loc["rank (without lattice)"] = rank_nolat.round(2)
    csv.to_csv(out / "table_uci.csv")
    dens.round(3).to_csv(out / "table_uci_precision.csv")

    # LaTeX
    def cell(d, c):
        m, s = full.loc[d, c], sd.loc[d, c]
        if not np.isfinite(mean.loc[d, c]):
            return "--"
        txt = f"{mean.loc[d, c]:.2f}"
        row = full.loc[d]
        if np.isfinite(m) and np.isclose(m, row.min()):
            txt = rf"\textbf{{{txt}}}"
        if c in AGENTS and complete.loc[d, c] < 5:
            txt += rf"$^{{[{int(complete.loc[d, c])}]}}$"
        txt += rf"\,{{\scriptsize({s:.2f})}}" if np.isfinite(s) else ""
        if c in AGENTS and flags.loc[d, c] > 0:
            txt += r"$^\dagger$"
        return txt
    spec = "l" + "c" * (len(INTERP) + len(BLACK) + 1) + ">{\\columncolor{agent}}c" * len(AGENTS) \
        + "r" * len(AGENTS)
    lines = [r"\begin{tabular}{@{}" + spec + "@{}}", r"\toprule",
             "dataset & " + " & ".join(cols) + " & " +
             " & ".join(rf"\#c\textsubscript{{{c}}}" for c in AGENTS) + r" \\", r"\midrule"]
    for d in idx:
        nc = [f"{nconst.loc[d, c]:.0f}" if np.isfinite(nconst.loc[d, c]) else "" for c in AGENTS]
        lines.append(d + (r"$^\ddagger$" if d in lattice else "") + " & "
                     + " & ".join(cell(d, c) for c in cols) + " & " + " & ".join(nc) + r" \\")
    lines += [r"\midrule", "rank (all) & " + " & ".join(f"{rank_all[c]:.2f}" for c in cols)
              + " & " * len(AGENTS) + r" \\",
              r"rank (no $\ddagger$) & " + " & ".join(f"{rank_nolat[c]:.2f}" for c in cols)
              + " & " * len(AGENTS) + r" \\",
              r"\bottomrule", r"\end{tabular}"]
    (out / "table_uci.tex").write_text("\n".join(lines) + "\n")

    # companion tables
    cost = pf.groupby("config").agg(runs=("dataset", "size"), minutes_median=("minutes", "median"),
                                    minutes_max=("minutes", "max"), turns_median=("turns", "median"),
                                    input_tokens_median=("input_tokens", "median"),
                                    output_tokens_median=("output_tokens", "median"),
                                    cost_usd_total=("cost_usd", "sum"))
    cost.reindex(AGENTS).to_csv(out / "table_cost.csv")
    sa = ok.groupby("config").agg(median_gap=("optimism_gap", "median"),
                                  n_abs_gap_below_0_1=("optimism_gap", lambda g: int((g.abs() < .1).sum())),
                                  median_constants=("n_literals", "median"),
                                  n_flagged=("flag", lambda f: int((f.fillna("") != "").sum())))
    sa.reindex(AGENTS).to_csv(out / "table_selfassess.csv")
    status = pf.groupby("config").status.apply(lambda s: int((s != "ok").sum()))

    num = {}
    for c in AGENTS:
        key = c.replace("-", "").replace("10", "Ten").replace("60", "Sixty").capitalize()
        num[f"{key}RankAll"] = f"{rank_all[c]:.2f}"
        num[f"{key}RankNoLattice"] = f"{rank_nolat[c]:.2f}"
        num[f"{key}NotScored"] = int(status.get(c, 0))
    (out / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n"
                                             for k, v in num.items()))
    print(csv.to_string())
    print(cost.to_string())
    print(sa.to_string())


if __name__ == "__main__":
    main()
