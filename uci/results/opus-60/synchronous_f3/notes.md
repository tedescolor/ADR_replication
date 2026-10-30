# Notes

## Findings
- x1 + x2 = 1 (up to noise of about 0.0015), so x1 adds nothing beyond x2.
- x3 has two regimes:
  - **Readout** (391/445 rows, x3 ≥ 36): y ≈ 1.1792 + 0.000998·x3 + 0.00036·x0. The residual sd is 0.00284, the residuals are Gaussian (kurtosis −0.26), and the noise does not depend on the covariates.
  - **Dropout** (54 rows): x3 ≈ N(0.7, 1.8²), with |x3| < 5. It carries no information about y (residual correlation 0.03). There is a clean gap between 4.9 and 36.9.
- In the dropout rows, y must come from (x0, x2). With a GP on (x0, x2) the noise floor is sd ≈ 0.046. The relationship is the same in both regimes, so the fallback is fitted on all 445 rows.

## Candidates (5-fold CV NLL)
For the fallback y | x0, x2 on all rows, I compared Normal, SkewNormal and Student-t. Mean: linear, quadratic, cubic or quartic polynomial. Log-scale: constant, linear or quadratic.
- The best options were cubic or quartic mean with linear or quadratic log-scale, at about −1.64 to −1.66. Skew and t gave no gain.
- I chose the 9-term cubic mean with quadratic log-sd (−1.643).

## Final program
`Mixture(w·N(readout mean, 0.002845), (1−w)·N(cubic(x0,x2), exp(quad(x0,x2))))`.
- w = 0.997·sigmoid((x3−20)/2). The switch sits in the empty gap.
- The 0.3% floor on the fallback guards against an outlier in the readout regime. It costs about 0.002 nats.
- 25 constants and 244 AST nodes.

## Validation
Full pipeline (refit both parts per fold), 5-fold CV repeated over 5 seeds:
- NLL −4.121 (sd across seeds 0.003); training NLL −4.131.
- Readout rows contribute about −4.45 each, dropout rows about −1.8. The score therefore depends mostly on the share of dropout rows in the test set (about 12% here), which adds sampling uncertainty.

expected_nll: -4.10
