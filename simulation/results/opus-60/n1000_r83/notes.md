# Conditional density estimation: notes

## Data
1000 rows, x0 ~ U(0,1), y in [-1.7, 2.6]. Scatter shows two crossing curves:
- "upper" branch: ~0 at x=0, minimum ~-0.8 near x=0.22, rising to ~2.5 at x=1;
- "lower" branch: emerges from the upper one near x~0.5, peaks ~0.1 near x=0.7, falls to ~-1 at x=1.
Noise decreases with x (sd ~0.33 at low x, ~0.1 at x>0.8). Within a branch the residuals are
roughly symmetric, slightly platykurtic (excess kurtosis ~ -0.5).

## Model family
Mixture of two Normals: polynomial means in t = x-0.5, log-polynomial scales, and a
logistic-polynomial weight for the lower branch. Fit by maximum likelihood (BFGS, several inits,
including one with a sharp weight switch and one with a gentle one; best train NLL kept).

## Validation
5-fold CV repeated twice (10 folds, fixed seeds, the same folds for every candidate, so paired
differences can be compared). Fold-level SE of the mean ~0.015; paired SE between neighbours ~0.0014.

cfg = (deg mean a, deg mean b, deg log-sd a, deg log-sd b, deg logit weight):
| cfg | #params | train | CV |
|---|---|---|---|
| (5,3,1,1,1) | 16 | 0.166 | 0.181 |
| (5,3,1,1,0) const weight | 15 | 0.175 | 0.189 |
| (5,3,1,1,2) | 17 | 0.164 | 0.178 |
| **(5,3,2,1,2)** | 18 | 0.160 | **0.1775** |
| (5,3,3,1,2) | 19 | 0.160 | 0.178 |
| (6,3,2,1,2) | 19 | 0.160 | 0.179 |
| (4,3,2,1,2) | 17 | 0.162 | 0.180 |
| (5,4,2,1,2) | 19 | 0.160 | 0.179 |
| (5,3,2,1,1) | 17 | 0.163 | 0.180 |
| (7,3,2,2,1), (8,4,2,2,1) | 21-24 | 0.161 | 0.185-0.192 |

Also tried, none helped in CV:
- sqrt(x) / log(x) basis terms for the means and scales (to catch the steep drop near x=0): 0.177-0.179;
- each branch as two equal-weight Normals at m ± k·s, to model the platykurtic noise (4 components):
  better train NLL but CV 0.179-0.181;
- Student-t was not tried seriously because the residuals are lighter-tailed, not heavier.

PIT check of the chosen fit: KS p=0.78, PIT z sd 1.00, no clear skew.

## Choice
(5,3,2,1,2): the best CV, though only by about 0.001 over its neighbours, which is within noise.
It is small (18 constants plus 0.5 and 1.0) and smooth. Refit on all data: train NLL 0.1604.

## Expected NLL
CV gives 0.1775 using 80% training folds. The final fit uses all the data, so it should do slightly
better, but the sampling uncertainty of the test set (SE ~0.015) is much larger than that effect.

expected_nll: 0.178
