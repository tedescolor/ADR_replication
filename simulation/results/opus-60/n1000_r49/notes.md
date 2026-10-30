# Notes

## Data
1000 rows, x0 in [0,1], y in [-1.6, 2.7]. Scatter plot: a single noisy curve for x < ~0.45
(dips to -0.85 at x~0.22), which then splits into two branches: an upper one that rises almost
linearly to ~2.6 at x=1, and a lower one that bends back down to ~-0.9. Noise shrinks with x
(sd ~0.35 at x=0, ~0.09 at x=1). About half the points are on each branch.

## Model family
Two-component Gaussian mixture where all parameters depend on x:
- lower mean m1(x): polynomial of degree 6 (degree 5 is tied);
- upper mean m2(x) = m1(x) + a * max(x - k, 0)^p (branches coincide below the knot k≈0.44);
- one shared sd: exp(quadratic in x);
- a constant weight (fitted value ≈ 0.485).

14 constants. Fitted by maximum likelihood (BFGS on the mean NLL) on all 1000 rows.

## Validation
5-fold CV, plus 10-fold CV repeated 3 times (paired per-row differences give SEs).
| model | params | CV NLL |
|---|---|---|
| free mixture: poly6 lower, poly3 upper, separate log-sd quad, logistic weight | 19 | 0.180 (5f) |
| same, constant weight, shared sd (B) | 15 | 0.174 (5f) |
| B with Student-t components | 17 | 0.174 (dfs go large, so no gain) |
| separate sds / logistic weight | 18-21 | 0.177-0.179 |
| **lower poly6 + power gap, shared sd quad, const weight (chosen)** | 14 | 0.1732 (5f), 0.1753 (10f x3) |
| poly5 + power gap | 13 | 0.1731 (5f), 0.1757 (10f x3) |
| softplus gap, poly5/6 | 13-14 | 0.1729 (5f), 0.1747-0.1763 (10f x3) |
| log-sd linear instead of quadratic | 12 | 0.1809 (10f x3), worse by 0.005-0.006 |
| sine-plus-linear lower curve | 12 | 0.180 |

The gap-parametrised variants differ by ≤0.0016, while the paired SE is ~0.002, so they are tied.
I kept poly6 + power gap. Standardised residuals by x-bin have mean ~0, sd ~1 and small
skewness/kurtosis. The only slight misfit is near the split (x≈0.56).
Training NLL is 0.160, which matches `adr.score.precision_nll`.

## Estimate
The CV estimates are 0.173-0.175. The final model is fit on all the data, so it should do
slightly better than the fold models.

expected_nll: 0.175
