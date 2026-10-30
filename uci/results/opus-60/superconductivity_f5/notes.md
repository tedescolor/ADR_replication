# Notes

## Data
17012 rows, 81 continuous covariates, y in [-0.78, 185]: strongly right-skewed and bimodal
(a large mass in 0-20, a second mode around 70-95), with small negative values (jitter around 0).

## What was tried (holdout = 5012 random rows, fit on the other 12000)
- Linear forward selection on sqrt(y): R2 about 0.70 with 10 covariates, 0.76 with all 81. Log-transforming
  covariates added little.
- Mixture of linear experts (Normal components; mean, log-scale and gate logits linear in the
  same 8 covariates): holdout NLL 3.78-3.83.
- **Projection mixture**: J linear projections s_j = w_j . x feed every head. Mixture weights are
  softmax(linear in [1, s1, s2]), component means are linear in [1, s1] (or [1, s1, s2]), and log-scales
  are linear in [1, s1]. Using all 81 covariates, holdout NLL was 3.70 (J=1) and 3.60 (J=2), but that
  needs about 200 constants.
- Sparsification to fit in 80 constants: an L1 (or group-L1) penalized fit on the training part ranks the
  individual projection weights. I kept the top N weights (a per-projection mask), refit without the penalty,
  and scored on the holdout:
  - J=1, 58 covariates: 3.707
  - J=2, 23 shared covariates: 3.722
  - J=2, K=4, mask 47, means on [1,s1,s2]: 3.698
  - J=2, K=4, mask 51, means on [1,s1]: **3.680**
  - J=2, K=5, mask 49, means on [1,s1], constant log-scale: 3.672
- Each projection is clipped to its observed training range (minimum/maximum), so the
  model stays valid and bounded anywhere in the covariate box.

## Validation and final choice
I validated the two best structures with the same 5-fold CV, refitting all parameters in each fold:
- K=4, mask 51 weights, means on [1,s1], log-scales on [1,s1]: folds 3.699/3.671/3.667/3.655/3.654, mean 3.669
- **K=5, mask 49 weights (34 in s1, 15 in s2), means on [1,s1], one constant log-scale per component**:
  folds 3.671/3.644/3.645/3.633/3.637, **mean 3.646 (se 0.006)**. It is better on every fold.

Final program: s1 and s2 are sparse linear projections of the raw covariates (49 weights across 44
covariates), each clipped to its observed range (4 constants). The weights are softmax(s1 - s1, 4 logits
linear in [1, s1, s2]), and the 5 Normal components have mean a_k + b_k*s1 and scale exp(c_k).
That is 80 literals and 468 AST nodes, and the program passes the contract check. The constants
come from a refit on all 17012 rows (in-sample NLL 3.639). The mask was chosen on a split that
overlaps the CV folds, so the CV figure is slightly optimistic. The full-data refit offsets this
a little, so my estimate is a bit above the CV mean.

expected_nll: 3.655
