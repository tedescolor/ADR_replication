"""Score the archived ADR programs on their test folds (deterministic; no agent call).

  PY scripts/score.py        # -> results/per_fold.csv, results/per_row.csv

Per run: test NLL (mean of -log p(y1, y2 | x) over the 200 test rows of the fold), train NLL,
the agent's expected_nll, counted constants and AST nodes, minutes, turns and cost.
A program that raises or returns a non-finite log-density on a test row is not scored on that
fold (status non-finite), as in the UCI study; the NLL over its finite rows is kept as a diagnostic.
"""
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONFIGS, FOLDS, RESULTS, result_dir, split  # noqa: E402
from adr.contract import check_source, count_literals, load_model  # noqa: E402


def nll(model, X, Y):
    """Per-row log-density; NaN where the program raises (e.g. a non-finite parameter)."""
    out = []
    for x, y in zip(X, Y):
        try:
            out.append(float(np.asarray(model(x).logpdf(y[None, :])).reshape(-1)[0]))
        except Exception:
            out.append(np.nan)
    return np.array(out)


def main():
    runs, rows = [], []
    for c in CONFIGS:
        for k in range(1, FOLDS + 1):
            d = result_dir(c, k)
            rec = {"config": c, "fold": k}
            if not (d / "model.py").exists():
                rec["status"] = "missing"
                runs.append(rec)
                continue
            src = (d / "model.py").read_text()
            tree = ast.parse(src)
            rec.update(constants=count_literals(tree), ast_nodes=len(list(ast.walk(tree))),
                       contract="ok" if not check_source(src) else "; ".join(check_source(src)))
            Xtr, Ytr, Xte, Yte = split(k)
            model = load_model(d / "model.py")
            lp_te, lp_tr = nll(model, Xte, Yte), nll(model, Xtr, Ytr)
            ok = np.isfinite(lp_te)
            rec.update(status="ok" if ok.all() else "non-finite",
                       test_nll=float(-lp_te.mean()) if ok.all() else np.nan,
                       n_failed_rows=int((~ok).sum()),
                       test_nll_finite_rows=float(-lp_te[ok].mean()),  # diagnostic only
                       train_nll=float(-np.nanmean(lp_tr)))
            rows += [{"config": c, "fold": k, "row": int(i), "logpdf": float(v)}
                     for i, v in zip(np.flatnonzero(_test_mask(k)), lp_te)]
            m = re.search(r"expected_nll:\s*([-+0-9.eE]+)", (d / "notes.md").read_text()) \
                if (d / "notes.md").exists() else None
            rec["expected_nll"] = float(m.group(1)) if m else np.nan
            rec["optimism_gap"] = rec["test_nll"] - rec["expected_nll"]
            if (d / "run_summary.json").exists():
                s = json.loads((d / "run_summary.json").read_text())
                rec.update(minutes=s.get("minutes"), turns=s.get("num_turns"),
                           cost_usd=round(s.get("cost_usd") or 0, 2), run_status=s.get("status"))
            runs.append(rec)
            print(f"{c} fold {k}: test NLL {rec['test_nll']:.4f} ({rec['status']}, "
                  f"{rec['n_failed_rows']} failed rows; finite rows {rec['test_nll_finite_rows']:.4f}), "
                  f"train {rec['train_nll']:.4f}, expected {rec['expected_nll']}, "
                  f"{rec['constants']} constants", flush=True)
    pd.DataFrame(runs).to_csv(RESULTS / "per_fold.csv", index=False)
    pd.DataFrame(rows).to_csv(RESULTS / "per_row.csv", index=False)


def _test_mask(k):
    from common import fold_of_rows
    return fold_of_rows() == k


if __name__ == "__main__":
    main()
