# Notes

## Data
1000 rows, x0 in [0,1], y continuous. Scatter plot: a single noisy curve (min about -0.8 near x=0.2)
that splits around x≈0.51 into two branches. The upper one rises to about 2.4 at x=1 and the lower one
bends down to about -1. The branches look equally populated. Noise is heteroscedastic, with sd going from
about 0.4 near x=0 to about 0.09 near x=1. Residual kurtosis is about 0, so Gaussian noise fits.

## Candidates (5-fold CV repeated twice = 10 folds, mean held-out NLL; full refit with 8 random restarts per fold)
- Free 2-component Gaussian mixture: cubic means, log-linear sds, logistic weight. CV 0.185 (one 5-fold repeat).
- Structured "trunk + split": m(x) is a polynomial of degree K in t=x-0.5, u=max(x-x0,0) with the
  split point x0 fitted, branch means m+g(u)±h(u) where g and h are polynomials in u of degree H
  with no constant term, a shared log-sd that is a polynomial of degree S in t, and a mixture weight.
  | K,H,S | CV NLL (se≈0.010) | train |
  |---|---|---|
  | 3,2,2 | **0.1765** | 0.1660 |
  | 4,2,1 | 0.1768 | 0.1668 |
  | 4,2,2 | 0.1774 | 0.1660 |
  | 5,2,2 | 0.1777 | 0.1660 |
  | 4,3,2 | 0.1787 | 0.1658 |
  | 4,2,3 | 0.1795 | 0.1650 |
  | 4,1,2 | 0.1875 | 0.1772 |
  | 5,2,2 + separate sds per branch | 0.1802 | 0.1661 |
  H=2 (curved branches) clearly helps. The other variations do not change CV beyond noise.

## Final
K=3, H=2, S=2, with the weight fixed at 0.5 (the fitted logit was 0.008). That gives 13 constants, fitted by
ML on the full data (train NLL 0.1660). I chose the smallest model with the best CV.

expected_nll: 0.178
