# Conditional density for y | x0

## Data
250 rows, x0 in (0,1). Scatter shows two branches:
- a curved "lower" branch over the whole x range (wavy: ~-0.2 at 0, min ~-0.9 near 0.25, ~0.2 near 0.6, ~-1 at 1);
- a tight linear "upper" branch y ~ -2.45 + 5.1 x that only appears for x > ~0.47
  (where binned kurtosis drops to about -2, i.e. bimodal).
Lower-branch noise is visibly larger for x < ~0.47 (sd ~0.34) than after (~0.15).

## Validation
10-fold CV repeated 3 times (30 fits per model), mean held-out NLL per row. Each fold re-fits
by ML (BFGS + Nelder-Mead), started from the full-data fit.

## Candidates (CV NLL)
| model | train | CV |
|---|---|---|
| 2-normal mixture, lower mean poly3, logsd const, logit-w linear | 0.271 | 0.315 |
| lower poly4, logsd linear (both comps), logit-w quadratic | 0.183 | 0.241 |
| lower poly4, logsd linear, w = cap * step(x > c), upper sd const | 0.191 | 0.233 |
| same, lower mean = a + bx + A sin(fx+phi) | 0.208 | 0.283 |
| same, lower mean poly5 | 0.191 | 0.240 |
| same, lower SkewNormal | 0.180 | 0.229 |
| **lower poly4, lower sd two-level step at same c, w = cap*step** | **0.180** | **0.224** |
| + skew in lower | 0.179 | 0.225 |
| + separate threshold for sd step | 0.170 | 0.227 |
| + linear logsd trend on top of step | 0.179 | 0.228 |

A freely fitted logistic weight went to an essentially infinite slope, so a hard step
(`where`) is used.

## Final program
Mixture of two Normals:
- lower: mean = quartic in x; sd = 0.335 for x <= 0.473, 0.146 above;
- upper: mean = -2.453 + 5.109 x, sd = 0.140;
- weight of upper = 0.577 for x > 0.473, else 0.
12 constants. Train NLL 0.180, CV NLL 0.224. I picked it because it had the best CV score
and the fewest constants of the good models.
Slightly optimistic because of model selection and because CV folds start from the full-data
threshold, so I add a small margin.

expected_nll: 0.235
