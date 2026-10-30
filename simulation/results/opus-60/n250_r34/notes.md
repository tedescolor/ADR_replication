# Notes

Data: 250 rows, x0 in (0,1). Scatter shows a single noisy curve (sd ~0.29) for x0 < ~0.62
that then forks into two thin branches (sd ~0.13-0.16): an upper branch rising linearly to
~2.5 at x0=1 and a lower branch falling to ~-0.95. Roughly 55-60% of points beyond the
fork lie on the lower branch.

Model: 2-component Gaussian mixture.
- A: cubic mean over all x0 (base curve, continuing into the upper branch), log-sd
  b0 + b1*s(x), with s(x) = sigmoid(k(x - c)) the "fork" switch (noise shrinks after fork).
- B: quadratic mean (lower branch), constant sd, weight sigmoid(wl) * s(x).
13 fitted constants, MLE with BFGS from several hand-chosen inits (the likelihood has bad
local optima where B becomes a broad background component).

Validation: 4 x repeated 5-fold CV, refitting everything (multi-start) per fold.
- polynomial A + linear B + logistic weight (first attempt, bad optimum): CV ~0.33-0.35
- linear log-sd, linear B: CV 0.322
- noise step tied to fork, linear B, hard switch: CV 0.277
- noise step tied to fork, quadratic B, smooth switch (FINAL): train 0.179, CV 0.245 (se ~0.056)
- separate switches for noise and weight: train 0.168, CV 0.262 (worse)
- quartic instead of cubic base mean: train 0.179, CV 0.251 (worse)

Final chosen for best CV. Expected held-out NLL slightly above CV mean to account for
selection and fold-level optimisation noise.

expected_nll: 0.26
