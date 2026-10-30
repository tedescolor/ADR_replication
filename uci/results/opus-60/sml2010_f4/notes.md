# Notes

Data: 2211 rows, 20 covariates (all jittered/continuous). x16-x18 look like pure noise (dropped);
x7, x8, x10, x12, x13 add nothing once other terms are in (dropped by backward elimination).

Validation: 5-fold CV on Gaussian NLL. Term selection used folds from seed 0; final candidates
were re-scored with independent fold splits (seeds 1 and 2), refitting all coefficients
(mean + log-scale) inside each fold.

Tried (CV mean NLL):
- constant Normal: 3.99; linear Gaussian (all x): 2.99; linear + squares: 2.94
- additive hinge GAM (homoscedastic): 2.75; small tanh MLP (heteroscedastic): 2.75
- parametric linear + tanh steps in x3/x4 + log-linear scale: 2.795
- greedy hinge selection (22 hinges) + log-linear scale: 2.64 (too big for the AST budget)
- greedy forward selection over hinges max(x_j - k, 0) and pairwise products x_a*x_b on top
  of a linear base: strong gains from a step in x3 (~195), interactions x5*x9, x6*x19, x1*x5,
  x6*x9, x11*x19, and hinges in x2, x5, x6, x0, x1.
  18 terms: 2.607 / 2.613; 24 terms: 2.560 / 2.562 (seeds 1 / 2).

Final program: Normal(m, s) with
- m = intercept + linear in x0,x1,x2,x3,x4,x5,x6,x9,x11,x14,x15,x19 + 13 products/squares
  + 11 hinge terms (written with dot() over grouped knots);
- log s = linear in x0, min(x3,230), x5, x6, x19, capped at 4.
64 constants, 483 AST nodes. Gaussian residuals were near-normal (slightly light-tailed), Student-t
did not help.

Expected NLL: CV ~2.56, plus an allowance (~0.03) for optimism from greedy selection on the same
data and the extrapolative product terms.

expected_nll: 2.59
