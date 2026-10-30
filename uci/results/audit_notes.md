# Audit notes

Opus 5.5 (`opus-60`), 60 tasks with the covariate domain, run on 26 October 2026 on the server described in
the top-level README, workspace `/ws/a`, 12 agents at a time (one fold at a time), Claude Code 2.1.309.
Harness probe: all 19 checks PASS (`results/probe_*.json`).

## Flags reviewed and kept

- **t0016** (NavalPropolsion, fold 1), flag `outside: /opt`: false positive. The text is a `sed` substitution
  `s/options={'maxiter':20000,...}/options={...}/` applied to a file in `work/`.

All other runs are clean. All 60 programs pass the contract check with `--domain` and give finite densities
at every test row. Contamination scan: 0 matches in the 60 transcripts (`results/contamination_scan.csv`).
