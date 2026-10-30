# Notes
Data: 250 rows, 1 covariate x0 in [0,1]. Mean is nonlinear in x0; for x0 > ~0.44 the
conditional density becomes bimodal (a lower branch and an upper branch rising to y~2.5).

Tried (10-fold CV mean NLL, same folds):
- single Normal, polynomial mean (deg 3-6) with polynomial log-sigma: best 0.644 (cubic mean, quadratic log sigma);
  Student-t and skew-normal gave no gain (residual kurtosis -1 flagged the mixture structure).
- two-component Normal mixture: lower branch cubic mean + log-linear sigma, upper branch linear mean + constant sigma,
  weight = cap * sigmoid(k (x - c)):
  cubic/linear-logsigma/linear-upper: CV 0.282 (final); other variants: quadratic upper mean 0.284, 0.299 with k allowed up to 400,
  quadratic log sigma 0.304, constant sigma 0.379, quartic mean 0.637, quadratic mean 0.496, 5th-degree 0.656.
Final: that mixture (12 fitted values, ~20 literals), fit on all data by MLE (L-BFGS-B, k bounded at 80). Train NLL 0.238.
Expected NLL = CV 0.282 plus a small allowance for model selection optimism and the hard-wired branch point.

expected_nll: 0.30
