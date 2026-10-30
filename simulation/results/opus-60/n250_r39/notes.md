# Notes

## Data
250 rows, one covariate x0 in (0,1). For small x0 there is a single noisy curve: y dips to about -0.8
near x0≈0.28, rises to about +0.3 near 0.65, then falls to about -1 at x0≈1. Its noise sd shrinks
from ~0.37 to ~0.12 as x0 grows. A second, nearly linear branch (y ≈ -2.7 + 5.5·x0, sd ≈ 0.15) is
clearly visible for x0 > 0.6 and reaches ~2.5 at x0≈1. At high x0, about 40-50% of the points lie on it.

## Models tried (2-component Gaussian mixture, ML fit with BFGS + random restarts)
Validation: 10-fold CV, repeated with 2 different shuffles. Each fold refits on 9/10 of the data with
multiple restarts. The table gives the mean held-out NLL.

| lower mean | upper mean | lower log-sd | gate | params | CV NLL |
|---|---|---|---|---|---|
| cubic | linear | const | logistic | 9 | 0.299 |
| cubic | linear | linear | logistic | 10 | 0.239 |
| quartic | linear | linear | logistic | 11 | 0.228 |
| quintic | linear | linear | logistic | 12 | 0.228 |
| quartic | quadratic | linear | logistic | 13 | 0.233 |
| quartic | cubic | linear | logistic | 14 | 0.228 |
| quartic + Student-t | quadratic | linear | logistic | 14 | 0.233 |
| quartic | quadratic | quadratic | logistic | 14 | 0.235 |
| quartic | quadratic, upper log-sd linear | linear | logistic | 14 | 0.264 |
| quartic | quadratic | linear | logistic × max weight | 14 | 0.229 |
| **quartic** | **linear** | **linear** | **logistic × max weight** | **13** | **0.224** |
| quintic | linear | linear | logistic × max weight | 14 | 0.227 |

Cubic is not flexible enough for the lower curve, and a linear log-sd is clearly needed. Student-t
tails do not help (the fitted df is about 15). The gate with a capped weight turns on sharply at
x0≈0.26 and plateaus at a weight of 0.39. So the upper line effectively starts below the lower curve and
crosses it, which explains a few low outliers near x0≈0.2-0.3.

## Final program
Mixture of two Normals:
- lower component: quartic mean, sd = exp(a + b·x0);
- upper component: linear mean, constant sd;
- weight = sigmoid(41.5·(x0-0.259))·sigmoid(-0.459).

It has 13 fitted constants (15 literals including the 1.0 in 1-w). Refit on all data, the training NLL is 0.157.
I chose it because it has the lowest CV NLL (0.224), and the gap to the other good variants is small.

## Expected NLL
The CV estimate is 0.224. I add a little for selecting among about 14 variants on the same CV and for the noisy
tails of a 250-row sample.

expected_nll: 0.235
