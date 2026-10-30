# Notes

## Data
7654 rows, 4 covariates, y in [420, 496]. A linear fit leaves residual sd 4.56. The residuals
are skewed left with heavy tails: the body looks normal, and a few very large negative outliers
reach -11 sd.

## Validation
5-fold CV, with one fixed random permutation, on mean held-out NLL in the original y units.
All models are fitted by direct maximum likelihood (L-BFGS, analytic gradients through the
network, 2-3 random restarts). Standard errors across folds are about 0.005-0.011.

## Results (CV NLL)
| model | #params | CV |
|---|---|---|
| linear mean, constant scale, Student-t | 7 | 2.927 |
| quadratic mean, log-linear scale, t | 21 | 2.850 |
| linear + 4 tanh units mean, log-linear scale, t | 35 | 2.820 |
| same, noise = 2-normal mixture (outlier comp.) | 37 | 2.809 |
| quadratic + 4 tanh, mixture | 47 | 2.800 |
| linear + 6 tanh, t | 47 | 2.801 |
| linear + 8 tanh, t | 59 | 2.793 |
| quadratic + 6 tanh, t | 57 | 2.796 |
| quadratic + 4 tanh mean + 2 tanh in log-scale, t | 57 | 2.794 |
| **linear + 10 tanh, log-linear scale, mixture** | 73 (+1 literal) | **2.777** (se 0.0065, train 2.757) |
| linear + 11 tanh, mixture | 79 (+1) | 2.777 (se 0.011) |

The mixture noise beats Student-t by about 0.01, and each extra pair of hidden units gains
about 0.005-0.01. Train/CV gaps grow from about 0.012 to 0.026 as H increases; overfitting is moderate.

## Final program
The mean is a linear term plus H tanh units on the raw covariates, with the standardization
folded into the weights. log(scale) is linear. The noise is Mixture([1, q], [N(m, s),
N(m + d*s, k*s)]), a small-weight wide outlier component.

I chose H=10 with mixture noise. H=11 gave no CV gain (2.7765 vs 2.7773), so I kept the smaller model. It uses
74 literals and 379 AST nodes. Writing each unit as `dot([..4..], x)` keeps the AST under 500.
The final model was refitted on all the data with 3 restarts; its train NLL is 2.758.
Across the box corners the scale stays positive (exp), so the density is always valid.

The estimate below is the CV mean. Refitting on all rows should help a little, and
generalization error and seed variance should hurt a little, so the two roughly cancel.

expected_nll: 2.78
