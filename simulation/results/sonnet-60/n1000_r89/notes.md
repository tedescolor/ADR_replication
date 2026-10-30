# Notes
- Data: 1 covariate, y has nonlinear mean and, for x>~0.6, a bimodal conditional (two branches, gap between ~0.3 and ~1.2).
- Single heteroscedastic Normal/StudentT/SkewNormal with polynomial mean and log-sd: 5-fold CV NLL ~0.60.
- Two-component Normal mixture, with polynomial means, log-linear sds, logistic-linear weight in x:
  5-fold CV NLL ~0.202-0.205 across many degree choices (repeated with 2 different fold splits).
- Final: mean degrees (comp A cubic, comp B quartic), log-sd linear in both, weight sigmoid-linear; 16 fitted constants (plus 2 for u=2x-1, 1 for 1-w), fitted by ML on all data.
  Chosen as the simplest config at the CV optimum. Train NLL 0.187, so the optimism is ~0.015.
- Components are flexible curves rather than strictly "branches", which is fine for density purposes.
expected_nll: 0.21
