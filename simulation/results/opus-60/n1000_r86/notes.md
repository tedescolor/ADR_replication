# Notes

Data: 1000 rows, x0 in [0,1]. Scatter shows a single noisy cloud (sd ~0.3) for x < ~0.45
that splits into two thin branches (sd ~0.1-0.15): one rising to ~2.5 at x=1, one falling to ~-1.
Branches carry roughly equal weight (~0.48/0.52).

Validation: 5-fold CV (fixed random folds), mean held-out NLL; refit on all data for final constants.
Multi-start BFGS for maximum likelihood.

Tried:
- Free 2-normal mixture, polynomial means (deg 3-5), log-sd (deg 1-2), logit weight (deg 1-2):
  CV 0.218-0.223 (best (4,2,2) = 0.218). Components overlap awkwardly at low x.
- Structured mixture: shared centre c(x) (quartic) +/- gap g(x) = k * softplus ramp starting near x0,
  so before the split both components coincide; separate quadratic log-sd per branch; constant weight.
  (4,1,2,0): CV 0.2137 (train 0.1969). Adding a quadratic gap term: 0.2141 (no gain).
  Residuals in each branch close to Gaussian (|skew|<0.3, small excess kurtosis), so no heavy tails.

Final: structured model (4,1,2,0), 15 constants. CV NLL 0.2137, train 0.197.
Expected NLL on new data ~ CV value (refit on 25% more data should help slightly).

expected_nll: 0.214
