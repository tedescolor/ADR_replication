# Notes

## Data
250 rows, one covariate x0 in (0,1). Scatter shows a single wavy curve for x0 < ~0.47 that
bifurcates: an upper branch rising nearly linearly to y~2.5 at x0=1, and a lower branch
continuing the wave (peak ~0.2 at x0~0.7, falling to ~-1 at x0=1). For x0 > 0.5 about 60-65%
of points lie on the upper branch, roughly independent of x0. Noise on the lower branch
shrinks with x0 and is right-skewed at small x0.

## Candidates (all 2-component mixtures, fitted by ML, Nelder-Mead + BFGS)
Validation: 5-fold CV repeated 3 times (15 folds, refit on each training split), mean held-out NLL.

| model | description | train NLL | CV NLL |
|---|---|---|---|
| V1 | Normal sinus+linear lower, quadratic upper, logistic gate | 0.105 | 0.187 |
| V2 | V1 with gate = q * sigmoid(step) (step + plateau) | 0.095 | 0.186 |
| V3 | V2 with SkewNormal lower branch | 0.081 | **0.166** |
| V4 | V2 with Student-t lower branch | 0.095 | 0.191 |
| V5 | V2 with linear upper mean, constant upper sd | 0.110 | 0.182 |
| V6 | V2 with log-quadratic lower sd | 0.095 | 0.198 |
| V7 | V3 with linear upper mean, constant upper sd | 0.098 | 0.172 |
| V8 | V7 with log-linear upper sd | 0.089 | 0.172 |
| V9 | V7 with skew shape linear in x0 | 0.098 | 0.174 |

Fold-to-fold standard error of the CV mean is ~0.04, so differences are modest, but the
skew-normal lower branch consistently helped.

## Final program (V3), 16 fitted constants
- lower: SkewNormal(loc = a + b sin(w x0 + phi) + c x0, scale = exp(s0 + s1 x0), shape)
- upper: Normal(u0 + u1 x0 + u2 x0^2, exp(t0 + t1 x0))
- weight of upper = sigmoid(q) * sigmoid(steep linear in x0): ~0 for x0 < 0.46, ~0.62 above.
At the switch point the upper branch mean coincides with the lower branch, so the hard gate
is not fragile.

CV NLL 0.166; adding a small allowance for model selection among ~8 variants:

expected_nll: 0.18
