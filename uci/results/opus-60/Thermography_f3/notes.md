# Notes

## Data findings
- x1..x24 are highly correlated level-type covariates (~35); x0 is a different kind of covariate.
  Rows look exchangeable (lag-1 residual autocorrelation ≈ 0), so I used shuffled K-fold CV.
- Key structure: D = x24 - y is almost never negative (min -0.02), with a sharp peak at
  D ≈ 0.01-0.08 and a long right tail. x24 also is >= max(x1..x23) - 0.024 in every row.
  So x24 behaves like max(y, x1..x23) + small noise. When the gap g = x24 - max(x1..x23)
  is clearly positive, y is usually the maximum, i.e. it sits just below x24. Otherwise y is
  a draw from a left-skewed distribution (effectively truncated at the max).

## Validation
10-fold CV repeated twice (20 fits), mean held-out NLL, with all parameters refit in each
fold (ridge penalty on the linear location coefficients).

| model | CV NLL |
|---|---|
| Normal, ridge-linear mean on all 25 | 0.603 |
| Student-t, linear mean | 0.572 |
| Student-t, log-scale linear in x0 and ensemble sd | 0.567 |
| 2-normal mixture, linear mean, heteroscedastic | 0.423 |
| Gated mixture: tight Normal at x24 - c + Normal(linear loc) | 0.291 |
| same, broad component SkewNormal | 0.209 (lam 1), 0.2025 (lam 5) |
| + tight component SkewNormal | 0.191 |
| + broad scale depends on tanh(20 g) | 0.193 |
| **both (final)** | **0.1896** |

## Final program (37 constants)
- loc = linear in x0..x24 (26 coefficients, 5 `dot` calls)
- g = x24 - max(x1..x23) (via `maximum.reduce`)
- gate w = sigmoid(a + b*tanh(k*g) + c*(x24 - loc))
- y ~ w * SkewNormal(x24 - c1, s1, shape1) + (1-w) * SkewNormal(loc, s*exp(d*tanh(20 g)), shape2)
Fitted by penalised MLE on all data (ridge lam = 5 on the slope coefficients).
In-sample NLL 0.149; CV 0.190. I chose it because it is the best CV model and every term is
motivated by the max-structure; the gain from the last two refinements is small (~0.013),
so I add a little allowance for selection optimism.

expected_nll: 0.20
