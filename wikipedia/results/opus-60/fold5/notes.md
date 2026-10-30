# Notes

## Data
800 rows, 6 covariates (x0 is a count from 1 to 51, x1 and x2 are heavy-tailed counts, x3 is binary, x4 is in [0, 25], x5 is heavy-tailed and positive). y0 > 0 and is close to lognormal. y1 > 0, with some values near 0, and is strongly related to y0 (corr 0.55).
Covariates enter as log x0, log(1+x1), log(1+x2), x3, x4, x4^2 and log x5. The x4^2 term was added because the log y0 residuals have a U shape in x4, and it improved CV by about 0.03.

## Factorisation
p(y0) p(y1|y0), using `Sequential`. The two factors have separate parameters, so each one is fitted
and validated on its own by MLE (L-BFGS, tiny ridge).

## Validation
5-fold CV, repeated with 2 random partitions (10 folds), mean held-out NLL per row.
Each number below is the CV NLL of that factor.

y0 (log scale + Jacobian):
- LogNormal, constant scale: 1.5457 (1.5744 without x4^2)
- LogNormal, log-scale linear in all features: 1.5247
- LogNormal, log-scale on log x0, x3, x4 only: **1.5227** (chosen)
- Gamma 1.5298; log-t 1.5247; log-skew-normal 1.5249; 2-component lognormal mixture 1.5332
- Squared and interaction terms: no gain

y1 | y0 (only adr.dist families, so a sqrt-normal (0.970) is not expressible):
- Normal, location on features plus y0: 0.992. Adding log y0 and a log-scale linear in y0: 0.9694
- SkewNormal 0.966, Laplace 0.975, Gamma 1.056, StudentT **0.9629**
- Log-scale depending on all features: worse (0.969, overfits)
- Squared and interaction terms: no gain
- **Mixture: StudentT (as above, ~95%) + a narrow Normal(0.20, 0.096) (~5%): 0.9399** (chosen).
  Letting the mixture weight depend on y0 gave only 0.9394, so it was dropped. Using a Gamma as the
  second component gave 0.9414.

Final configuration, re-run as one joint CV: y0 1.5227 + y1|y0 0.9399 = **2.4626**
(standard error over folds ≈ 0.023). In-sample NLL with the full-data fit is 2.4267.

## Final program
Fitted on all 800 rows, with coefficients folded into raw-covariate form and rounded to 4
significant figures (in-sample NLL unchanged). It has 36 constants. The log-scales are clipped
so the model stays valid anywhere in the covariate box. The contract check passes.

## Expected NLL
CV gives 2.4626. Fitting on all the data should help a little, and selecting among many
variants should hurt a little, so I estimate about 2.465.

expected_nll: 2.465
