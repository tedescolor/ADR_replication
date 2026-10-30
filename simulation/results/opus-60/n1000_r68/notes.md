# Notes

## Data
1000 rows, x0 ~ U(0,1). Scatter shows one noisy curve for x < ~0.55 (Gaussian residuals,
sd ~0.35 at x=0 shrinking with x) that splits into two branches for x > 0.6
(upper rising to ~2.6, lower falling to ~-1), roughly 50/50 in mass, with small noise (~0.1).

## Candidates (2-component Gaussian mixtures, MLE with BFGS, multiple restarts)
Validation: 5-fold CV repeated 3x with different shuffles (per-row out-of-fold NLL averaged),
paired comparisons between models on the same rows.
Notation (mean poly degree, sigma poly degree, shared sigma, weight degree; -1 = 0.5/0.5).

| model | params | CV NLL |
|---|---|---|
| 3,0,shared,equal (constant sigma) | 10 | 0.267 |
| 3,1(log-linear) separate, logistic-linear w | 14 | 0.188-0.192 |
| 3,1 shared linear, equal w | 10 | 0.189 |
| 3,1 separate linear, equal w | 12 | 0.191 |
| 3,1 shared, constant w | 11 | 0.190 |
| **3,1 shared linear, logistic-linear w** | **12** | **0.1839** |
| 4,1 shared, equal / const / linear w | 12-14 | 0.185-0.187 |
| 5,1 shared, equal / linear w | 14-16 | 0.187 |
| 4,2 shared, linear w | 15 | 0.186 |
| common curve +/- relu split (S3/S4) | 9-10 | 0.192-0.196 |

Heteroscedasticity is essential (constant sigma is +0.08 worse). Beyond that, differences
are a few thousandths (paired se ~0.002-0.004); the chosen model is best and smallest
among the top group. Higher degrees overfit slightly.

## Final program
Mixture of two Normals: cubic means m1(x), m2(x); shared sd = |0.3642 - 0.2756 x| (+0.001
floor), i.e. 0.36 at x=0 to 0.09 at x=1; weight of the upper branch sigmoid(1.639 - 2.053 x).
12 fitted parameters (14 literals). Refit on all data: train NLL 0.1715.

CV NLL 0.184; standard error of a mean over new rows ~0.023 per 1000 rows.

expected_nll: 0.185
