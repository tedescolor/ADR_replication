# Notes

Data: 250 rows, x0 in [0,1]. Scatter shows a single wavy branch for x < ~0.5
(sd ~0.3) that splits for x > ~0.55 into an upper, roughly linear branch (y ~ -2.1 + 4.8x,
small noise) and a lower sine-like branch falling to ~-1 at x=1. Roughly 45% of points
sit on the upper branch past the split.

Model family: 2-component normal mixture.
- comp 1 mean: a*sin(b*x+c) + d + e*x; log-sd linear in x
- comp 2 mean: linear in x; log-sd linear in x
- weight of comp 2: sigmoid(k(x - c0)), optionally multiplied by a cap wmax.

Validation: repeated 5-fold CV (2-3 repeats), refitting all parameters by ML in each fold
(Nelder-Mead + BFGS, multiple starts).
- base (uncapped sigmoid weight): train 0.259, CV 0.313
- + quadratic log-sd for comp 1: CV 0.318 (worse)
- + quadratic upper mean: CV 0.330 (worse)
- capped weight (wmax~0.47, onset step at x~0.33): train 0.240, CV 0.301 (3x5-fold;
  same-run comparison base vs cap: 0.314 vs 0.301)

Final: capped-weight mixture, step steepness fixed at 50 (ML pushed it to a hard step;
50 is a smooth equivalent, train NLL 0.244). 16 constants.
Estimated held-out NLL: CV ~0.30 plus a small allowance for selection optimism.

expected_nll: 0.31
