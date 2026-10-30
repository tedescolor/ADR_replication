# Notes

## Data
800 rows, 6 skewed covariates (x0 counts 1..51, x1 0..1865, x2 0..83 with many zeros,
x3 binary, x4 0..25, x5 0.04..164), y0 > 0.9 right-skewed, y1 > 0 (some mass near 0).
Covariate transforms used: a=log x0, b=log(1+x1), c=log(1+x2), x3, g=x4/10, h=log x5,
plus q=(g-1.4)^2 (U-shaped residual in x4), centred squares/products a2, h2, ah.

## Model family
Factorisation p(y0|x) p(y1|y0,x) via `Sequential`:
- y0 ~ LogNormal(m0(x), exp(s0(x))) — log y0 residuals are close to Gaussian (skew 0.07).
- y1 | y0 ~ SkewNormal(m1(x, y0), exp(s1(x, y0)), alpha) with log y0 and y0 both in the mean
  (the y1 mean is convex in y0) and log y0 in the log-scale.
All predictors are linear in the transformed features. Fit by maximum likelihood (L-BFGS,
analytic gradients).

## Tried
- MVN / Normal y0 (CV ~2.55), SkewNormal y0 (2.52) — clearly worse than LogNormal (2.487).
- y1|y0: Normal 2.487, SkewNormal 2.482, Student-t 2.488, Gamma 2.565, 2-component normal
  mixture (gain only 0.002, not worth constants).
- quadratic x4 term: -0.024. Adding y0 (in addition to log y0) to y1 mean: -0.018.
- Greedy stepwise add/drop of terms in each of the 4 predictors (pool: all main effects,
  centred squares, a few products, c0 = [x2==0], y0, (log y0)^2), scored by 4x repeated 5-fold CV,
  penalising additions by 0.001. Took CV from 2.464 to 2.421.

## Validation
5-fold CV, repeated with 4 fold seeds for selection; the final candidates re-scored on 10
*fresh* fold seeds not used during selection:
- baseline (35 params): 2.482
- stepwise intermediate (38 params): 2.431
- stepwise after 10 steps (34 params): 2.4224
- final (after 12 steps; 32 fitted coefficients incl. alpha, 43 literals incl. centring constants, 360 AST nodes): 2.4210 (sd over fold seeds 0.005)
Training NLL of final fit: 2.371.
Final predictors: m0: a,c,x3,g,h,q,ah,h2; s0: a,c,x3,g,h,h2; m1: a,b,c,g,h,q,log y0,y0,a2; s1: a,g,h,log y0.
Stepwise was stopped at 46 min for time; the last steps changed CV by ~0.001 each.

## Choice and estimate
Final = the stepwise-selected spec, refit on all 800 rows. The fresh-fold CV of 2.421 uses
640-row training sets (slightly pessimistic) but the spec itself was selected on CV (slightly
optimistic); also held-out noise. Estimate:

expected_nll: 2.43
