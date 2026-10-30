# Notes

## Data
1000 rows, x0 in [0,1]. Scatter shows a single noisy curve for x < ~0.42 that splits into two
branches: an upper one rising almost linearly to ~2.4 at x=1, and a lower one continuing the
curve and falling to ~-1 at x=1. Noise sd shrinks with x (~0.3 at x=0 to ~0.08 at x=1).
Standardized residuals look Gaussian (skew ~0, excess kurtosis ~0; Student-t fit drove df -> inf).

## Validation
5-fold CV (fold seed 1, repeated with seed 7 for the top candidates), maximum likelihood with
BFGS + 6 random restarts per fit, held-out mean NLL.

| model | CV NLL |
|---|---|
| 2-comp mixture, free quintic means, quad log-scales, logistic weight (20 par) | 0.173 |
| shared curve g deg5 + branch g + r(d0+d1 r), r=relu(x-c), quad log-scale (13) | 0.166 |
| same, g deg6, cubic log-scale (15) | **0.160** (seed 7: 0.159) |
| g deg5 / deg7, cubic log-scale | 0.162 / 0.161 (seed 7: 0.162 / 0.159) |
| quartic log-scale | 0.161 |
| branch-specific scale offset | no gain |
| linear-only divergence | 0.172 |
| Student-t noise | 0.160 (df -> inf) |

## Final program
Mixture of two Gaussians with a common scale exp(cubic in t=2x-1):
lower = g(x) (degree-6 polynomial in t), upper = g(x) + r*(d0 + d1 r) with r = max(x - c, 0),
c ~ 0.42 fitted; constant weight sigmoid(w) ~ 0.475 for the upper branch. 15 fitted parameters,
19 literals. Chosen as the best CV with fewest parameters among the near-ties; refitted on all data
(train NLL 0.137).

expected_nll: 0.16
