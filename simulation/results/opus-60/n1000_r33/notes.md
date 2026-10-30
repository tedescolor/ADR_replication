# Notes

## Data
1000 rows, x0 in [0,1]. For x < ~0.55 y follows one noisy curve (min ≈ -0.8 near x=0.22). Beyond that
it splits into two branches: one rising roughly linearly to ≈2.5 at x=1, the other falling to ≈-1.
Noise shrinks as x grows (sd ≈0.4 near x=0, ≈0.1 near x=1). At high x the within-bin kurtosis is about -2, i.e. bimodal.

## Candidates (maximum likelihood with BFGS, 2 starts)
1. Two-component Gaussian mixture. Both means are polynomials in x, log-sd is a polynomial in x,
   and the weight is sigmoid(poly). I tried mean degrees 3–7, log-sd degrees 1–3, shared vs separate
   sd, and a constant vs linear weight.
2. A structured "branching" mixture: base polynomial plus a branch offset in powers of a
   softplus ramp after a learned split point. CV ≈0.188–0.191, no better than (1), so I dropped it.

PIT histograms of the polynomial mixture were flat in every x-octile, so it is well calibrated.

## Validation
5-fold CV repeated 4 times (20 folds, same splits for every model). I compared models using
paired fold differences.
- deg4 means, deg2 log-sd shared, constant weight: **0.1901** (reference)
- deg4, sd deg2, linear weight: +0.0018 (se 0.0008)
- deg4, separate sds: +0.0028 (se 0.0006)
- deg4, sd deg1 / deg3: +0.0038 / +0.0032
- deg5/6 means: +0.0036 to +0.0050
- deg3 means: +0.013

## Final
The final model is a two-component Gaussian mixture. Each component mean is a degree-4 polynomial
in x, both share sd = exp(quadratic in x), and the weight is constant (≈0.5). That is 14 fitted
constants, refit on all 1000 rows. Training NLL is 0.171 and the CV NLL (fits on 80% of the data)
is 0.190. Fitting on the full data should help slightly, so my estimate is a little below the CV value.

expected_nll: 0.188
