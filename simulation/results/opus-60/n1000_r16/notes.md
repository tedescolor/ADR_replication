# Notes

## Data
1000 rows, x0 in (0,1), y in about (-1.6, 2.7). A scatter plot (work/scatter.png, work/fit.png) shows two parts:
- a smooth curve over the whole range: it dips to about -0.75 at x≈0.25, rises to about 0.15 at x≈0.65 and falls to -1 at x=1. Its noise shrinks with x (sd ≈ 0.37 near x=0 and ≈ 0.08 near x=1);
- a nearly straight rising branch (y ≈ -1.1 at x≈0.32 up to 2.5 at x=1). It is absent for x < ~0.315, it crosses the curve near x≈0.45, and it holds about 46% of the mass for x > 0.315.

## Model family
A 2-component Normal mixture, with `t = 2x-1`:
- curve: polynomial mean in t, log-sd quadratic in t;
- branch: quadratic mean, log-sd linear in t;
- branch weight = c * sigmoid(k (x - x0)), where c is constant and k is large, so it acts as a step.
Everything is fitted by maximum likelihood (BFGS, multiple starts).

## Validation
5-fold CV repeated twice (10 fits per configuration, different splits), reporting the mean held-out NLL.
The whole model is refitted in each fold, including the step location.

| config (mean-curve deg, branch deg, curve log-sd deg, branch log-sd deg, weight) | #par | CV NLL |
|---|---|---|
| 6,2,1,1 plain sigmoid weight | 16 | 0.278 |
| 6,2,1,1 Student-t curve | 17 | 0.361 (optimisation unstable) |
| 6,2,1,1 logit-quadratic weight | 17 | 0.248 |
| 6,2,1,1 capped-step weight (c*sigmoid) | 17 | 0.241 |
| 6,2,2,1 capped step | 18 | 0.240 |
| 6,2,2,1 step * logit-linear weight | 19 | 0.242 |
| 5,2,2,1 capped step | 17 | 0.237 |
| **4,2,2,1 capped step** | **16** | **0.2352** |
| 3,2,2,1 capped step | 15 | 0.248 |
| 4,1,2,1 / 4,2,2,0 / 5,2,3,1 / 5,2,2,2 / 5,3,2,1 | | 0.249 / 0.252 / 0.238 / 0.239 / 0.240 |

The standard error of the mean NLL is about 0.025, but the differences between configurations are paired and much less noisy.
The plain-sigmoid weight did markedly worse. That model made the branch component leak in as a heavy tail at low x, which is the wrong structure.

## Final choice
4,2,2,1 with the capped-step weight: 16 fitted constants, about 30 literals in total.
It has the best CV, it is the most parsimonious of the top models, and it matches the visual structure.
The step steepness is capped at 500 to keep the program numerically benign. It fits to 0.2129 on the full data, and the exported file scores 0.2135.

## Expected NLL
The CV estimate is 0.235. I add a little for selection optimism; the full-data fit uses 25% more data than each fold's fit, which partly offsets that.

expected_nll: 0.238
