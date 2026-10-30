# Notes

## Data
250 rows, x0 in [0,1]. For x0 < ~0.55, y is unimodal Gaussian-like (residual sd ~0.3,
skew ~0, kurtosis ~0) around a wavy mean (-0.45 near 0, -0.8 near 0.25, rising after).
For x0 > ~0.6, y splits into two narrow branches (sd ~0.1): upper rising to ~2.5 at x=1,
lower falling to ~-0.7 at x=1, roughly 55/45 split.

## Models tried (10-fold CV, 2 repeats unless noted; NLL per row)
- Generic 2-Normal mixture, poly means / log-sd / logit weight in x:
  (deg 3,1,1) train 0.194, CV 0.255; (4,1,1) CV 0.270; (5,1,1) CV 0.283 (overfits).
- Piecewise "split at t" hinge model: unstable optimisation, CV 0.78; dropped.
- Structured "branching" mixture (z = 2x-1): shared centre c(z) (poly), branch offsets
  m1 = c + exp(g1(z)), m2 = c - exp(g2(z)) (g quadratic, so offsets are ~0 at low x and
  grow smoothly), shared sd exp(poly), constant weight:
  - c deg 3, sd deg 1: CV 0.273
  - c deg 4, sd deg 1: CV 0.257
  - c deg 4, sd deg 2: train 0.181, CV 0.2525  <- chosen
  - same but separate sd per branch: CV 0.263 (overfits)
- c deg 5: train 0.170, CV 0.263 (10-fold, 1 repeat) -> overfits
- linear logit weight: train 0.181 (no gain), CV 0.261 (1 repeat) -> not needed

CV caveat: CV fits were warm-started from the full-data optimum (plus random restarts),
which may bias CV slightly optimistic.

## Final
Branching mixture with c deg 4, offsets quadratic, log-sd quadratic, constant weight:
15 fitted constants (+2 for z = 2x-1). Train NLL 0.181, CV 0.2525. Most loss is in the
0.6-0.8 transition zone where branch membership is ambiguous.
Expected NLL on new data: CV value plus a small allowance for warm-start optimism.

expected_nll: 0.26
