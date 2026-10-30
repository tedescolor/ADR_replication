# Notes

## Data
7654 rows, y ~ 454 +- 17. Linear OLS residual sd 4.55 (Gaussian NLL 2.935). Residuals are
heavy-tailed (excess kurtosis ~5.5) with a long left tail, and both scale and skew depend on
the covariates (strong negative skew at low x0 / high predicted y). A 1-NN difference estimate
(sd ~3.1) suggested the mean has more nonlinear structure than a cubic polynomial (CV sd 4.15).

## Models (all fitted by maximum likelihood, L-BFGS with analytic gradients, inputs standardised
during fitting, then folded back into raw-x constants)
Mean = linear(x) + sum of H tanh(linear(x)) units; noise = K-component normal mixture where
component k>0 has mean mu + linear(x), every component has log-sd linear(x), mixture logits linear(x).

5-fold CV (fixed random folds) mean NLL:
| H | K | constants | CV NLL |
|---|---|---|---|
| 0 | 1 | 10 | 2.933 |
| 0 | 2 | 25 | 2.833 |
| 0 | 3 | 40 | 2.801 |
| 4 | 1 | 34 | 2.835 |
| 4 | 2 | 49 | 2.790 |
| 4 | 3 | 64 | 2.774 |
| 6 | 2 | 61 | 2.769 |
| 8 | 2 | 73 | 2.759 (l2=1e-3: 2.770) |
| 6 | 3 | 76 | 2.755 (l2=1e-3 on hidden weights: 2.754) |

The mixture (shape/tails) matters more than the mean; both help. Chosen: H=6, K=3 with small L2
on hidden-layer weights (best CV, fits in the 80-constant budget). Final fit on all data, best of
4 random initialisations by training NLL (train NLL 2.733). Exported model.py reproduces it
exactly (2.7329 on data.csv); checked over 2000 random points in the domain box that all
component sds are finite and positive. Fold-to-fold CV spread is ~0.02.

Expected held-out NLL: CV estimate 2.754 (trained on 80% of the data); the full-data fit should be
similar or marginally better, so I report ~2.755 allowing for a small optimism from model selection.

expected_nll: 2.755
