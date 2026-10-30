# Notes

## Data
1000 rows, x0 in [0,1]. Scatter: a single noisy curve (dips to about -0.8 near x=0.25, rises after that)
that forks at x ≈ 0.46. The upper branch keeps going up (to 2.5 at x=1). The lower branch bends
back down (to -1 at x=1). The two branches carry roughly equal mass. Noise sd shrinks with x
(≈0.38 at x=0 down to ≈0.09 at x=1). Standardised residuals look Gaussian (skew≈0, excess kurtosis≈0).

## Validation
5-fold CV repeated 4 times (20 folds, fixed seeds, the same folds for every candidate), mean held-out NLL.
Fold-to-fold SE of the mean ≈0.01; paired differences between nested variants have SE ≈0.002.
(Small caveat: in the fork-model CV, each fold's optimiser was also started from the full-data fit
as well as from a generic init. Only the starting point is shared, but this could make the CV
slightly optimistic.)

## Candidates (CV NLL)
- Free 2-normal mixture: polynomial means, log-polynomial sds, logistic weight linear in x:
  0.226–0.231 (degree 5–8 for the lower mean).
- Same, with the logit quadratic in x: 0.209 (17 params).
- "Fork" model: up = base polynomial; lo = up - b*max(x-c,0)^p; shared sd exp(poly);
  constant weight. Before the fork the two components coincide.
  - lower = up - b*r (linear divergence): 0.214
  - power divergence p≈1.52, log-linear sd, base deg 4/5/6: 0.191 / 0.192 / 0.193
  - power, log-quadratic sd, base deg 5 / 4 / 3: 0.1909 / 0.1895 / **0.1885**
  - power + an extra r^2 term: 0.195 (worse)

## Final
Base cubic, fork at c=0.461, divergence 8.93*r^1.519, sd = exp(quadratic), weight ≈0.508.
11 fitted parameters (12 literals including the 0.0 in the max). Train NLL 0.1773, CV NLL 0.1885.
I chose it because it is the simplest model and has the best CV. Allowing for a little
selection optimism, I estimate the NLL on new data at about 0.19.

expected_nll: 0.190
