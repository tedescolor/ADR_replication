# Audit notes

Opus 5.5, five folds, run on 26 October 2026 on the server described in the top-level README, workspace
`/ws/b`, one agent at a time, Claude Code 2.1.309, with the covariate domain of the full dataset in each task.
Harness probe: all 19 checks PASS (`results/probe_*.json`).

All five runs are clean in the audit. Every fold is scored: the programs give finite densities at all 200 test
rows of each fold.

Contamination scan: one match of the term `categories` in fold 4, which is the pandas attribute in
`pd.qcut(y0, 8).categories` used by the agent to bin a response; not a reference to the data. No other matches.
