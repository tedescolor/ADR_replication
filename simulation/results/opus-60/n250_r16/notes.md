# Notes

## Data
250 rows, x0 in [0,1], y in [-1.5, 2.5]. Scatter plot: a smooth wavy curve (min ~-0.85 at x~0.2,
max ~0.3 at x~0.7, -0.9 at x=1) with heteroscedastic noise (sd ~0.45 near x=0, ~0.1 near x=1),
plus a second, very tight linear branch (y ~ 0.06 + 5.23 (x-0.5), sd ~0.11) that appears
above x~0.5. The branch's share grows with x and is about 50% near x=0.8-1.

## Model family
2-component Normal mixture:
- base: mean = quartic in t=x-0.5, log sd = quadratic in t;
- branch: linear mean, constant sd;
- weight of the branch: linear in x, clipped to [0.001, 0.999] (it reaches 0 at x~0.16).
13 fitted constants, 21 literals in total. Fitted by maximum likelihood (BFGS + Nelder-Mead, several starts).

## Validation
5-fold CV repeated 2-3 times with different splits; the model was refit on each training fold.
Mean held-out NLL:
| variant | CV NLL |
|---|---|
| quartic base, sigmoid weight | 0.228 |
| cubic / quintic / sextic base, sigmoid weight | 0.27 / 0.234 / 0.239 |
| sine base, sigmoid weight | 0.249 |
| Student-t(5) base | 0.333 |
| constant weight | 0.332 |
| **quartic base, linear weight (floor 1e-3)** | **0.2148** |
| same, floor 1e-4 / 1e-2 | 0.2147 / 0.2163 |
| + linear log-sd for branch | 0.223 |
| + cubic log-sd for base | 0.226 |
| quintic base, linear weight | 0.222 |
| log-linear base sd | 0.2175 |

I chose the one with the lowest CV score; the extra flexibility in the other variants did not help.
Training NLL with all the data is 0.161; the CV score (0.215, from models fit on 200 rows) is
the honest estimate. A model refit on all 250 rows should do slightly better, but I add a margin for
the randomness of the test set.

expected_nll: 0.22
