# Audit summary

How the runs were audited: `tools/audit.py` scans every transcript for web-tool calls, network commands and
paths outside the task directory (`results/audit.csv`); every flag was reviewed by hand. An integrity breach
discards the run, which is repeated in a new task directory; a conduct slip (a third concurrent process or a
second Python interpreter) is recorded and the run kept.

| config | runs kept | integrity breaches | conduct slips | runs repeated for agent conduct or output (reason) |
|---|---|---|---|---|
| sonnet-60 | 200 | 0 | 11 | 1 (s154: the run ended without delivering a program) |
| opus-60 | 200 | 0 | 7 | 0 |

- Accepted limitation: the home path `/home/user` appears inside the harness (sandbox rules, environment
  variables the agents are not shown). The workspace path shown to the agents (`/ws`) is neutral.
- Harness probe: all checks PASS. Metric validation: `results/validation.json`.
- Contamination scan (`results/contamination_scan.csv`): CPFN, push-forward, the authors, KCDE, the arXiv id,
  `qs_true`, `branched`: no matches in the 400 kept transcripts.
- Library exposure: the snapshot given to the agents contained `adr/score.py` (the precision-aware scorer of
  the UCI study). `TASK.md` did not mention it; its text appears in 40 of the 400 transcripts, and those runs
  score the same as the others (see README, Section 7).
- Known limitations: the cost table counts resumed sessions; the `turns` field is unreliable for some runs.
