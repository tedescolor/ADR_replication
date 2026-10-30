"""Table 2 of the CPFN paper with the agents' rows added.

  PY scripts/make_tables.py   ->  results/tables/{table2.csv, table2.tex, table_extra.csv, table_cost.csv}

Published rows are copied from published/table2.csv (a single run of CPFN and of KCDE, no standard
deviations). Agent rows are the AWD and W1 at the three covariates of each scored run; with more than
one run per config the mean (sd, ddof = 1) is given. results/validation.json gives the AWD of the
true model, the floor of the metric with 1000 samples per x.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONFIGS, PKG, RESULTS  # noqa: E402

COLS = ["AWD", "W1_x_ring", "W1_x_trans", "W1_x_blobs"]
LABEL = {"sonnet-60": "ADR, Sonnet 5.5, 60 min", "opus-60": "ADR, Opus 5.5, 60 min"}


def cell(v):
    v = v.dropna()
    if len(v) > 1:
        return f"{v.mean():.3f} ({v.std(ddof=1):.3f})"
    return f"{v.iloc[0]:.3f}" if len(v) else "--"


def main():
    out = RESULTS / "tables"
    out.mkdir(parents=True, exist_ok=True)
    pub = pd.read_csv(PKG / "published" / "table2.csv")
    pub["runs"] = 1
    pr = pd.read_csv(RESULTS / "per_run.csv")
    for c in COLS + ["excess_nll", "test_nll", "optimism_gap", "n_literals", "minutes", "turns",
                     "input_tokens", "output_tokens", "cost_usd"]:
        if c not in pr:
            pr[c] = np.nan
    ok = pr[pr.status == "ok"]
    rows = []
    for c in CONFIGS:
        g = ok[ok.config == c]
        rows.append({"method": LABEL[c], **{m: cell(g[m]) for m in COLS}, "runs": len(g)})
    table = pd.concat([pub.rename(columns={"method": "method"}), pd.DataFrame(rows)], ignore_index=True)
    val = RESULTS / "validation.json"
    if val.exists():
        v = json.loads(val.read_text())
        floor = {"method": "true model (floor, 1000 vs 1000 samples)", "AWD": f"{v['floor']['AWD']:.3f}",
                 "W1_x_ring": f"{v['floor']['W1_x_ring']:.3f}", "W1_x_trans": f"{v['floor']['W1_x_trans']:.3f}",
                 "W1_x_blobs": f"{v['floor']['W1_x_blobs']:.3f}", "runs": ""}
        table = pd.concat([table, pd.DataFrame([floor])], ignore_index=True)
    table.to_csv(out / "table2.csv", index=False)
    lines = [r"\begin{tabular}{@{}lrcccc@{}}", r"\toprule",
             r"Method & runs & AWD & $W_1|x_\mathrm{ring}$ & $W_1|x_\mathrm{trans}$ & $W_1|x_\mathrm{blobs}$ \\",
             r"\midrule"]
    for _, r in table.iterrows():
        pre = r"\rowcolor{agent}" if str(r["method"]).startswith("ADR") else ""
        lines.append(pre + f"{r['method']} & {r['runs']} & " + " & ".join(str(r[m]) for m in COLS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table2.tex").write_text("\n".join(lines) + "\n")
    extra = ok.groupby("config").agg(runs=("rep", "size"), excess_nll=("excess_nll", "mean"),
                                     test_nll=("test_nll", "mean"), median_constants=("n_literals", "median"),
                                     median_optimism_gap=("optimism_gap", "median"))
    extra.to_csv(out / "table_extra.csv")
    cost = pr.groupby("config").agg(runs=("rep", "size"), not_scored=("status", lambda s: int((s != "ok").sum())),
                                    minutes_median=("minutes", "median"), turns_median=("turns", "median"),
                                    input_tokens_median=("input_tokens", "median"),
                                    output_tokens_median=("output_tokens", "median"), cost_usd_total=("cost_usd", "sum"))
    cost.reindex(list(CONFIGS)).to_csv(out / "table_cost.csv")
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
    print(extra.round(4).to_string())
    print(cost.to_string())


if __name__ == "__main__":
    main()
