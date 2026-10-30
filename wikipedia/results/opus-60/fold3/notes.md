# Notes

## Data
800 rows, 6 covariates (x0 count-like, x1/x2 skewed counts, x3 binary, x4 in [0,25], x5 skewed positive).
y0 > 0 is roughly log-normal; y1 > 0 is right-skewed and gamma/Weibull-like, with some values near 0.
corr(y0, y1) = 0.55. Features: log x0, log(1+x1), log(1+x2), x3, x4, log x5 (raw x1/x2/x5 did worse).

## Validation
5-fold CV repeated 3 times (15 folds, same splits for every candidate so the comparisons are paired).
Metric: mean joint NLL of (y0, y1) on held-out folds. The fold SE is about 0.026, but paired differences
between candidates are much less noisy than that.

## Candidates (CV joint NLL)
- y0 marginal: LogNormal with heteroscedastic scale (log-linear) beats Normal, Gamma, log-t and log-skew-normal.
  Adding x4^2 to the log-mean helps (the effect of x4 is U-shaped).
- y1 | y0: Weibull with heteroscedastic scale fit best (0.966), but Weibull is not in adr.dist. SkewNormal with
  linear location (in features + y0) and log-linear scale is next best (~0.966 when using raw y0). Gamma did worse (1.06),
  and so did Normal and Student-t.
- Base Sequential(LogNormal, SkewNormal): 2.4853. Adding y0^2 to the y1 location: 2.4811.
- Reversed order (y1 first) was about 0.005 better, but Sequential always conditions y1 on y0, so I dropped it.
- Log-y0 residuals were platykurtic (kurtosis -0.38) with a bimodal PIT, which suggests a latent class.
  A 2-component mixture with shared slopes, a shifted y0 intercept and feature-dependent logistic gating: 2.4747.
  As a joint mixture, Mixture of 2 Sequentials, where the class also shifts the y1 location and log-scale: 2.4720.
  Adding a separate y0 scale and a separate y1 shape per class gave no gain.
- L2 penalty on the parameters (penalised mean NLL): reg 1e-4 -> 2.4720, 2e-3 -> 2.4660, 5e-3 -> 2.4650, 1.2e-2 -> 2.4708.

## Final program
Mixture of 2 components, with weight w = sigmoid(linear in the 6 features).
Each component is Sequential(LogNormal(mu + d0*[class b], sd), SkewNormal(loc(x, y0, y0^2) + d1*[b], exp(s(x, y0) + e1*[b]), k)).
mu is linear in the features plus x4^2; log sd is linear in the features.
The model has 51 constants and 473 AST nodes. It was fit on all data with reg = 5e-3; the in-sample NLL is 2.395.
CV NLL is 2.465. The regularisation strength and the structure were chosen on the same CV, so a little
selection optimism applies.

expected_nll: 2.475
