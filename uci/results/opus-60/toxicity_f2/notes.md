# Notes

## Data
726 rows, 6 covariates. x3 and x4 are jittered counts (mostly 0, up to 4 / 6). y > 0, roughly
symmetric with heavy tails. A linear OLS mean explains most of the mean; squares and interactions
reduce CV MSE only a little. Residuals are clearly heteroscedastic (larger at extreme fitted values,
at small/large x0 and x5) and heavy-tailed (excess kurtosis about 3).

## Validation
10-fold CV of the mean held-out NLL, every parameter refit on the training folds by maximum likelihood (BFGS).
Model selection used repeated 10-fold splits (seed 1). The final model was re-scored on new random
splits (seed 7, 3 repeats).

## What I tried (CV NLL)
- Normal, const sd: 1.357. Student-t: 1.309. Laplace: 1.305.
- t with log-scale linear in x: 1.278. Laplace hetero: 1.287. Normal hetero: 1.336.
- Skew-normal, 2-normal mixture, normal+t mixture (same mean/scale): no better than t (1.223-1.230 vs 1.221).
- sqrt(y) and Box-Cox transforms: no better than the identity (lambda was about 0.9).
- t with all squares, or with cubes, in the mean and scale: 1.23-1.30 (overfits).
- Greedy forward selection (t family) over squares, interactions, sqrt(x3), sqrt(x4), x3>0.5 and x4>0.5
  for the mean and the log-scale. Order of additions: scale x0*x5 (1.230), mean x0*x5 (1.208),
  mean x0*x1 (1.194), mean sqrt(x3) (1.184), mean x1^2 (1.175), mean 1[x4>0.5] (1.170),
  scale x1*x5 (1.167). The search then stopped because no addition gained 0.002.

## Final program
Student-t (df 5.75). Location: linear in x0..x5 plus (x0-2.9)(x5-2.1), (x0-2.9)(x1-0.6),
sqrt(max(x3,0)), (x1-0.6)^2 and 1[x4>0.5]. Log-scale: linear plus (x0-2.9)(x5-2.1) and
(x1-0.6)(x5-2.1). The program has 29 constants. Train NLL is 1.132. CV NLL on fresh splits is 1.163.
The selection was done on CV, so I add about 0.01 for optimism.

expected_nll: 1.175
