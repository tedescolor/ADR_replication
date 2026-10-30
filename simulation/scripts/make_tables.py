"""Table 1 of the CPFN paper with the agents' rows added, and companion tables.

  PY scripts/make_tables.py   ->  results/tables/{table1.csv, table1.tex, table_extra.csv,
                                                  table_cost.csv, numbers.tex}

Published rows are copied from published/table1.csv (mean (sd) over 100 replicates). Agent rows
are mean (sd, ddof = 1) over the scored replicates, as in the notebook (errors.std(ddof = 1)).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONFIGS, PKG, QUANTILE_LEVELS, RESULTS, SIZES  # noqa: E402

METRICS = ["AWD"] + [f"AQE_{q:.2f}" for q in QUANTILE_LEVELS]
LABEL = {"sonnet-60": "ADR, Sonnet 5.5, 60 min",
         "opus-60": "ADR, Opus 5.5, 60 min"}


def cell(v):
    v = v.dropna()
    return f"{v.mean():.3f} ({v.std(ddof=1):.3f})" if len(v) > 1 else (
        f"{v.mean():.3f}" if len(v) else "--")


def main():
    out = RESULTS / "tables"
    out.mkdir(parents=True, exist_ok=True)
    pub = pd.read_csv(PKG / "published" / "table1.csv")
    pub = pub[pub.n.isin(SIZES)].reset_index(drop=True)  # only the sizes analysed here
    pr = pd.read_csv(RESULTS / "per_rep.csv")
    for col in METRICS + ["excess_nll", "test_nll", "optimism_gap", "n_literals", "minutes",
                          "turns", "input_tokens", "output_tokens", "cost_usd"]:
        if col not in pr:
            pr[col] = np.nan
    ok = pr[pr.status == "ok"]
    rows = []
    for c in CONFIGS:
        for n in SIZES:
            g = ok[(ok.config == c) & (ok.n == n)]
            rows.append({"method": LABEL[c], "n": n, **{m: cell(g[m]) for m in METRICS},
                         "replicates": len(g)})
    agents = pd.DataFrame(rows)
    pub["replicates"] = 100
    table = pd.concat([pub, agents], ignore_index=True)
    table.to_csv(out / "table1.csv", index=False)

    lines = [r"\begin{tabular}{@{}llrcccccc@{}}", r"\toprule",
             r"Method & $n$ & reps & AWD & AQE(0.10) & AQE(0.25) & AQE(0.50) & AQE(0.75) & AQE(0.90) \\",
             r"\midrule"]
    for i, r in table.iterrows():
        if i and r["method"] != table.loc[i - 1, "method"]:
            lines.append(r"\midrule")
        name = r["method"] if (i == 0 or r["method"] != table.loc[i - 1, "method"]) else ""
        cells = [str(r[m]).replace(" (", r"\,{\scriptsize(").replace(")", ")}") for m in METRICS]
        pre = r"\rowcolor{agent}" if r["method"].startswith("ADR") else ""
        lines.append(pre + f"{name} & {r['n']} & {r['replicates']} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table1.tex").write_text("\n".join(lines) + "\n")

    extra = ok.groupby(["config", "n"]).agg(
        replicates=("rep", "size"), excess_nll_mean=("excess_nll", "mean"),
        excess_nll_sd=("excess_nll", "std"), test_nll_mean=("test_nll", "mean"),
        median_constants=("n_literals", "median"), median_optimism_gap=("optimism_gap", "median"))
    extra.to_csv(out / "table_extra.csv")
    cost = pr.groupby("config").agg(
        runs=("rep", "size"), not_scored=("status", lambda s: int((s != "ok").sum())),
        minutes_median=("minutes", "median"), turns_median=("turns", "median"),
        input_tokens_median=("input_tokens", "median"),
        output_tokens_median=("output_tokens", "median"), cost_usd_total=("cost_usd", "sum"))
    cost.reindex(list(CONFIGS)).to_csv(out / "table_cost.csv")
    num = {}
    for c in CONFIGS:
        for n in SIZES:
            g = ok[(ok.config == c) & (ok.n == n)]
            key = c.replace("-", "").replace("10", "Ten").replace("60", "Sixty").capitalize()
            num[f"{key}AWD{n}"] = f"{g.AWD.mean():.3f}" if len(g) else "--"
    (out / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n"
                                             for k, v in num.items()))
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
    print(extra.round(4).to_string())
    print(cost.to_string())


if __name__ == "__main__":
    main()
