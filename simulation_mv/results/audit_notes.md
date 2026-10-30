# Audit summary

How the runs were audited: `tools/audit.py` scans every transcript for web-tool calls, network commands and
paths outside the task directory (`results/audit.csv`), and every tool call of both transcripts was also reviewed
by hand.

| config | runs kept | integrity breaches | conduct slips | repeated runs |
|---|---|---|---|---|
| sonnet-60 | 1 | 0 | 0 | 0 |
| opus-60 | 1 | 0 | 1 (a third concurrent process for about 15 s) | 0 |

- sonnet-60 (s001) notes, not treated as integrity breaches: an attempt to call a tool it does not have
  (`Monitor`, rejected by the harness); `ps aux | grep` on its own job; an import probe of torch, jax, autograd,
  sklearn and statsmodels with the given Python.
- Both agents read only inside their task directory and their own background-job logs (allowed).
- Both stopped on their own inside the budget: opus 12.3 min (44 turns, $1.40), sonnet 33.3 min (33 turns, $0.66).
- Accepted limitation: the home path `/home/user` appears inside the harness (sandbox rules, environment
  variables the agents are not shown). It is not in any task path or in `TASK.md`.
- Harness probe: all 19 checks PASS. Metric validation: `results/validation.json`
  (all 6 checks PASS; floor AWD 0.171; KCDE AWD recomputed 0.6843).
- Contamination scan (`results/contamination_scan.csv`): 0 matches in the 2 transcripts (benchmark terms and
  generator terms). Most reasoning blocks of the headless streams are recorded empty, so hidden reasoning is only
  partly searchable.
- W1 columns use the three covariates of the authors' R code and may differ in construction from the printed ones.

## Repeated Opus 5.5 runs (runs 1-4, 26 October 2026)

- Four further Opus 5.5 runs on the same training set, to measure the variability due to the agent
  (s004, s006, s008, s010), run on the machine of the first run, in the same workspace `/ws` with the same
  Python environment, two agents at a time. Harness probe: all 19 checks PASS (`results/probe_p20261026144543.json`).
- Audit: 4/4 clean. The agents read only inside their task directory and their own background-job logs (allowed).
- All four stopped on their own inside the budget: 48.1, 37.5, 41.9 and 18.2 min.
- Contamination scan: one match of the generator term `rot2` in run 2, which is the name of the agent's own
  script `work/rot2.py` (a rotation analysis); not a contamination. No other matches.
