# Notes

## Data
1000 rows, x0 in [0,1], y in [-1.5, 2.7]. Scatter plot: y follows one noisy curve for x below 0.5 (dips to about -0.7 at x near 0.25). Above x of about 0.5 it splits into two branches. One rises to about 2.5 at x=1, the other falls to about -1 at x=1. The noise sd shrinks as x grows, from about 0.34 at x=0 to about 0.09 at x=1. In the high-x bins the kurtosis is about -2, which confirms the two branches.

## Model family
A 2-component Normal mixture. Component means are polynomials in x, the log-sd is a polynomial in x, and the mixture weight is sigmoid(polynomial in x). The model is fitted by maximum likelihood (BFGS with several restarts, initialised from rough branch curves).

## Validation
5-fold CV repeated over 2-3 random partitions; every fold is refitted from scratch. Candidates were compared with paired per-row differences.

| mean deg / log-sd deg / weight deg | sd shared? | #par | CV NLL |
|---|---|---|---|
| 3/1/0 | no | 13 | 0.236 |
| 4/2/0 | no | 17 | 0.231 |
| 3/2/1 | no | 16 | 0.227 |
| 4/2/1 | no | 18 | 0.231 |
| 3/2/1 + Student-t | no | 18 | 0.230 |
| **3/2/1** | **yes** | **13** | **0.221-0.222** |
| 3/3/1 | yes | 14 | 0.221 (paired diff -0.0006 ± 0.002) |
| 2/2/1 | yes | 11 | 0.405 (quadratic means too rigid) |
| 3/1/1 | yes | 12 | 0.229 (+0.007 ± 0.004) |
| 3/2/0 | yes | 12 | 0.226 (+0.004 ± 0.004) |
| 4/2/1, 5/2/1 | yes | 15/17 | 0.223 / 0.227 |
| 3/2/1 + Student-t | yes | 15 | 0.221 |

When the two components had separate sds, the fitted sds were almost equal wherever the branches can be told apart. Sharing one sd improved CV, so the final model uses a shared sd. It has cubic means, a quadratic log-sd and a logistic-linear weight: 13 fitted constants, plus the integer exponents 2 and 3. Adding more terms gave no reliable gain.

## Final
`model.py` is the 3/2/1 shared-sd mixture refitted on all data (train NLL 0.207). The repeated 5-fold CV NLL is 0.222, with standard error about 0.023 across rows.

expected_nll: 0.222
