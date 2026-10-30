#!/usr/bin/env python3
"""Build paper tables from released CSVs, with explicit incomplete-fold handling."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
INPUTS = set()
CONFIGS = {"sonnet-60": "ADR (Sonnet 5.5)", "opus-60": "ADR (Opus 5.5)"}
DATASETS = ["energy", "synchronous", "localization", "toxicity", "concrete", "slump",
            "forestfires", "NavalPropolsion", "sml2010", "Thermography", "support2", "superconductivity"]
NAMES = {"NavalPropolsion": "Naval", "superconductivity": "Superconductivity",
         "forestfires": "Forest fires", "support2": "Support2", "sml2010": "SML2010"}
SUMMARY = {}


def read(path, **kw):
    path = ROOT / path
    INPUTS.add(path)
    return pd.read_csv(path, **kw)


def name(s):
    return NAMES.get(s, s.capitalize())


def cellmean(s):
    return float(str(s).split("(")[0])


def fmt(x, digits=3):
    return f"{x.mean():.{digits}f} ({x.std(ddof=1):.{digits}f})"


def write_table(stem, caption, label, headers, rows, note, small=True, col=None):
    col = col or "l" + "r" * (len(headers) - 1)
    body = [r"\begin{table}[tbp]", r"\centering", r"\caption{"+caption+"}",
            r"\label{"+label+"}", r"\footnotesize" if small else r"\small",
            r"\setlength{\tabcolsep}{4pt}", r"\begin{adjustbox}{max width=\textwidth}",
            r"\begin{tabular}{@{}"+col+r"@{}}", r"\toprule", " & ".join(headers)+r" \\", r"\midrule"]
    for row in rows:
        body.append(row if isinstance(row, str) else " & ".join(map(str, row))+r" \\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\par\smallskip",
             r"\begin{minipage}{\textwidth}\footnotesize "+note+r"\end{minipage}", r"\end{table}"]
    (OUT / (stem+".tex")).write_text("\n".join(body)+"\n")


def univariate():
    agg = read("simulation/results/tables/table1.csv")
    raw = read("simulation/results/per_rep.csv")
    assert len(raw) == 400 and raw.status.eq("ok").all()
    methods = agg.method.unique().tolist()
    metrics = ["AWD", "AQE_0.10", "AQE_0.25", "AQE_0.50", "AQE_0.75", "AQE_0.90"]
    labelmap = {"CPFN": "CPFN", "np.kde (KCDE)": "KCDE", "linear QR": "Linear QR",
                "qeML QR (nearest-neighbour)": "Nearest-neighbour QR", "fields QR (spline)": "Spline QR",
                "qrf QR (forests)": "Forest QR"}
    rows = []
    for n in [250, 1000]:
        if rows:
            rows.append(r"\midrule")
        rows.append(r"\multicolumn{7}{l}{\textit{$n="+str(n)+r"$}} \\")
        df = agg[agg.n == n].copy()
        best = {metric: df[metric].map(cellmean).min() for metric in metrics}
        for _, r in df.iterrows():
            m = r["method"]
            if "Sonnet" in m:
                label, conf = "ADR (Sonnet 5.5)", "sonnet-60"
            elif "Opus" in m:
                label, conf = "ADR (Opus 5.5)", "opus-60"
            else:
                label, conf = labelmap.get(m, m), None
            cells = []
            for metric in metrics:
                value = r[metric]
                if conf:
                    x = raw.loc[(raw.n == n) & (raw.config == conf), metric]
                    assert len(x) == 100 and abs(x.mean()-cellmean(value)) <= .00050001
                    assert fmt(x) == value, (m, n, metric, fmt(x), value)
                cells.append(r"\tabbf{"+value+"}" if cellmean(value) == best[metric] else value)
            rows.append([label]+cells)
    write_table("univariate", "Univariate simulation: mean error (Monte Carlo standard deviation) over 100 replicates.",
                "tab:univariate", ["Method", "AWD", r"AQE$_{.10}$", r"AQE$_{.25}$", r"AQE$_{.50}$", r"AQE$_{.75}$", r"AQE$_{.90}$"], rows,
                "Lower is better. Bold identifies the smallest displayed mean within each sample size and metric. "
                "Non-ADR entries are the archived CPFN-study summaries; ADR entries are verified against all 400 released runs. "
                "All entries use the Monte Carlo reference quantiles of the original study and 1,000 draws per covariate. "
                "The reported AWD integrates a truncated quantile grid, and the median AQE is particularly sensitive to Monte Carlo error between modes. "
                "ADR errors computed with exact quantiles are given in Table~\\ref{tab:uni-exact}.")
    rows = []
    for n in [250, 1000]:
        for c, label in CONFIGS.items():
            df = raw[(raw.n == n) & (raw.config == c)]
            rows.append([label, n, fmt(df.excess_nll), f"{df.n_literals.median():g}"])
    write_table("univariate_nll", "Univariate simulation: excess test NLL and program size.", "tab:uni-nll",
                ["Method", "$n$", "Excess NLL (sd)", "Median literals"], rows,
                "Excess NLL is the fitted minus true negative log-density on 10,000 independent test observations per replicate. "
                "Parentheses summarize variation across the 100 replicates. Counted literals follow the output contract.", False)
    SUMMARY["univariate"] = {f"{c}_{n}": {"AWD": float(df.AWD.mean()), "excess_nll": float(df.excess_nll.mean()),
                                             "literals_median": float(df.n_literals.median())}
                              for (c,n), df in raw.groupby(["config", "n"])}
    return raw


def univariate_exact():
    raw = read("simulation/results/per_rep_exact.csv")
    assert len(raw) == 400
    metrics = ["AWD", "AQE_0.10", "AQE_0.25", "AQE_0.50", "AQE_0.75", "AQE_0.90"]
    rows = []
    for n in [250, 1000]:
        if rows:
            rows.append(r"\midrule")
        rows.append(r"\multicolumn{7}{l}{\textit{$n="+str(n)+r"$}} \\")
        for c, label in CONFIGS.items():
            df = raw[(raw.n == n) & (raw.config == c)]
            assert len(df) == 100
            rows.append([label]+[fmt(df[m]) for m in metrics])
    write_table("univariate_exact", "Univariate simulation: ADR errors with exact quantiles.", "tab:uni-exact",
                ["Method", "AWD", r"AQE$_{.10}$", r"AQE$_{.25}$", r"AQE$_{.50}$", r"AQE$_{.75}$", r"AQE$_{.90}$"], rows,
                "Mean error (standard deviation) over 100 replicates, with the grid, quantile levels and quadrature of "
                "Table~\\ref{tab:univariate}. True quantiles are obtained from the analytic distribution function and fitted "
                "quantiles from the distribution function of each program, in place of the Monte Carlo reference and of "
                "1,000 draws per covariate. These values are not comparable with the other rows of Table~\\ref{tab:univariate}.")
    SUMMARY["univariate_exact"] = {f"{c}_{n}": {m: float(df[m].mean()) for m in metrics}
                                   for (c, n), df in raw.groupby(["config", "n"])}
    return raw


def bivariate():
    # Table 2 of the CPFN paper with two added rows: AWD and W1 at the three named covariates.
    raw = read("simulation_mv/results/per_run.csv")
    pub = read("simulation_mv/published/table2.csv", index_col="method")
    # rep 0 is the run of the main comparison; Opus 5.5 reps 1-4 repeat it on the same training sample
    assert len(raw) == 6 and raw.status.eq("ok").all()
    assert sorted(raw[raw.config == "opus-60"].rep) == [0, 1, 2, 3, 4]
    cols = ["AWD", "W1_x_ring", "W1_x_trans", "W1_x_blobs"]
    rows = [[m]+[f"{pub.loc[m, k]:.3f}" for k in cols] for m in ["CPFN", "KCDE"]]
    for c, label in CONFIGS.items():
        r = raw[(raw.config == c) & (raw.rep == 0)].iloc[0]
        rows.append([label + (", first run" if c == "opus-60" else "")]+[f"{r[k]:.3f}" for k in cols])
    opus = raw[raw.config == "opus-60"]
    rows.append(["ADR (Opus 5.5), median of five runs"]+[f"{opus[k].median():.3f}" for k in cols])
    assert abs(raw[(raw.config == "opus-60") & (raw.rep == 0)].AWD.iloc[0] - .193) < .0005
    SUMMARY["bivariate_repeats"] = {k: [round(float(v), 3) for v in opus.sort_values("rep")[k]] for k in cols + ["test_nll"]}
    write_table("bivariate", "Bivariate simulation: AWD and $W_1$ at three representative covariates.", "tab:bivariate",
                ["Method", "AWD", r"$W_1\mid X=x_{\mathrm{ring}}$", r"$W_1\mid X=x_{\mathrm{trans}}$",
                 r"$W_1\mid X=x_{\mathrm{blobs}}$"], rows,
                "One fitted model per method in the first four rows, with 5,000 training observations; the last row is the "
                "median over the first run and four repetitions on the same sample. AWD averages empirical Wasserstein "
                "distances at 30 test covariates; the $W_1$ columns use 1,000 true and 1,000 generated draws at each named "
                "covariate. CPFN and KCDE rows are as printed by \\citet{franco2025}.", False)
    return raw


def uci():
    raw = read("uci/results/per_fold.csv")
    pub = read("uci/published/table3_nll.csv", index_col="dataset").loc[DATASETS]
    cpfn = read("uci/published/cpfn_uci.csv", index_col="dataset").loc[DATASETS]
    assert len(raw) == 60 and raw.status.eq("ok").all() and set(raw.config) == {"opus-60"}
    bad = raw[~raw.status.eq("ok")]
    values = pub.map(cellmean)
    strings = pub.map(lambda s: f"{cellmean(s):.3f} ({float(re.search(r'\((.*?)\)', s).group(1)):.2f})")
    values["CPFN"] = cpfn.CPFN
    strings["CPFN"] = cpfn.CPFN.map(lambda v: f"{v:.3f}")
    for c in ["opus-60"]:  # the real-data studies use Opus 5.5 only
        for ds in DATASETS:
            g = raw[(raw.config == c) & (raw.dataset == ds) & raw.status.eq("ok")]
            count = len(g)
            assert count == 5
            strings.loc[ds, c] = fmt(g.test_nll, 3).split(" (")[0] + f" ({g.test_nll.std(ddof=1):.2f})" + (r"$^{*}$" if count == 4 else "")
            values.loc[ds, c] = g.test_nll.mean() if count == 5 else np.nan
    ranked = values
    assert ranked.notna().all().all() and len(ranked) == 12
    # One table as in the CPFN paper: means at the two decimals of the published baselines.
    # Top three per row by values rounded to two decimals; ties share a place and are all marked.
    # Mean ranks use the same rounded values, with tied methods given their average rank.
    rounded = values.round(2)
    ranks = rounded.rank(axis=1, method="average").mean()
    place = rounded.rank(axis=1, method="min")
    wins = {ds: rounded.columns[place.loc[ds] == 1].tolist() for ds in DATASETS}
    columns = ["CADET", "CART-h", "CART-k", "CKDE", "LSCDE", "NKDE", "CDTree",
               "LinCDE", "MDN", "NF", "CPFN", "opus-60"]
    rows = []
    for ds in DATASETS:
        cells = []
        for c in columns:
            value = f"{values.loc[ds, c]:.2f}"
            if place.loc[ds, c] == 1:
                value = r"\underline{\tabbf{"+value+"}}"
            elif place.loc[ds, c] <= 3:
                value = r"\tabbf{"+value+"}"
            cells.append(value)
        rows.append([name(ds)]+cells)
    rows.append(r"\midrule")
    rows.append(["Mean rank"]+[f"{ranks[c]:.3f}" for c in columns])
    labels = {c: CONFIGS.get(c, "KCDE" if c == "CKDE" else c) for c in columns}
    headers = ["Dataset"]+[labels[c] for c in columns]
    write_table("uci", "UCI benchmark: test density NLL.", "tab:uci", headers, rows,
                "Entries are means over five folds. For each dataset, the best value is underlined in bold and the second "
                "and third best are in bold, comparing values rounded to the two decimals of the published baselines; "
                "tied methods are all marked. Mean ranks over the 12 datasets use the same rounded values, with tied methods given their average rank. KCDE is labelled CKDE in the original benchmark source.")
    dims = []
    for ds in DATASETS:
        df = read(f"uci/data/uci/{ds}.csv")
        dims.append([name(ds), len(df), len(df.columns)-1])
    half = (len(dims)+1)//2  # two side-by-side panels, as in the CPFN paper
    rows = [left+right for left, right in zip(dims[:half], dims[half:]+[["", "", ""]])]
    write_table("datasets", "Benchmark datasets and dimensions.", "tab:datasets", ["Dataset", "$N$", "$d$"]*2, rows,
                "$N$ is the released sample size before splitting into five test blocks; $d$ is the number of predictors. "
                "The spelling of Naval is shortened from the source identifier NavalPropolsion.", False,
                col=r"lrr@{\hspace{2.5em}}lrr")
    SUMMARY["uci"] = {"ranked_methods": list(ranked.columns), "mean_ranks_12": ranks.to_dict(),
                        "best_displayed": wins,
                        "failed_run": bad[["config", "dataset", "fold", "status"]].to_dict("records")}
    values.to_csv(OUT / "uci_means_for_ranking.csv")
    return raw


def runtime(uni, mv, uci):
    rows = []
    for study, df in [("Univariate", uni), ("Bivariate", mv), ("UCI", uci)]:
        for c, label in CONFIGS.items():
            g = df[df.config == c]
            if g.empty:  # the UCI study uses Opus 5.5 only
                continue
            rows.append({"study": study, "agent": label.removeprefix("ADR (").removesuffix(")"),
                         "scored": int(g.status.eq("ok").sum()), "retained": len(g),
                         "median_minutes": round(float(g.minutes.median()), 2),
                         "median_literals": float(g.n_literals.median()),
                         "recorded_usd": round(float(g.cost_usd.sum()), 2)})
    # Quoted in the text only (no table): medians and archived charges per study and agent.
    SUMMARY["resources"] = rows


def main():
    a, b, c = univariate(), bivariate(), uci()
    univariate_exact()
    runtime(a, b, c)
    SUMMARY["inputs_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(INPUTS)}
    (OUT / "table_manifest.json").write_text(json.dumps(SUMMARY, indent=2)+"\n")
    print(json.dumps({k:v for k,v in SUMMARY.items() if k != "inputs_sha256"}, indent=2))


if __name__ == "__main__":
    main()
