# Notes

Target y is positive (2.3..82.5) and right-skewed; modelling log y with a Gaussian
(i.e. a LogNormal for y, so the density is on the original scale) beat modelling y directly.

## Tried (10-fold CV mean NLL, original y scale)
- linear mean in x0..x6, log x7, (log x7)^2, homoscedastic: 3.41 (y), 3.48 (log y)
- + heteroscedastic log-sd (linear): 3.35 / 3.41
- + a few hand-made features (water/binder ratio, x0*log x7, ...): ~3.26-3.28
- physically motivated nonlinear model (effective binder/water power law, saturating
  age curve with mix-dependent rate): 3.42 — worse, dropped
- full quadratic in (x0..x6, log x7) for log-y mean, ridge, linear log-sd: 3.166 (too many constants)
- forward selection (LOO-OLS criterion on log y) over ~75 candidate terms (pairwise products,
  log age powers and interactions, ratios to binder sum, log ratios, indicators x1>3, x2>3, x4>1),
  then heteroscedastic Gaussian fit by ML.

## Validation
Nested 10-fold CV: selection was rerun inside each training fold, so the
estimate is honest about the selection step.
Number of terms -> nested CV NLL: 15: 3.215, 20: 3.167, 25: 3.150, 30: 3.148, 35: 3.152,
40: 3.142, 45: 3.144, 50: 3.158. Beyond about 25 terms the curve is flat. I chose 30 terms
because they fit the AST-node budget.
(Non-nested CV with selection on all data gave 3.104 for 30 terms; that number is optimistic.)

## Residual distribution
With the 30 terms fixed, 10-fold CV gave: Gaussian on log y 3.113, Student-t on log y 3.093
(not available as a density on y), and a two-component scale mixture of LogNormals with the
same mu, sd ratio exp(k), 3.100. The mixture is 0.013 better than the Gaussian, so I used it.

## Final program
log y ~ w N(mu, sigma) + (1-w) N(mu, sigma*exp(k)), returned as a Mixture of two LogNormals
(w=0.581, exp(k)=2.07).
mu: intercept + 30 selected terms (log x0, w/b ratio, log(w/b)*log t, age polynomials and
interactions, indicators). log sigma: linear in log t, (log t)^2, x0, x4, x1, x2, clipped to the
training range +-0.5. Fitted on all 824 rows. In-sample NLL is 3.021.
46 constants, 491 AST nodes. Passes the contract check.

Estimate: nested 30-term Gaussian 3.148, minus the 0.013 mixture gain, about 3.135.
I round it up slightly for the remaining optimism.

expected_nll: 3.14
