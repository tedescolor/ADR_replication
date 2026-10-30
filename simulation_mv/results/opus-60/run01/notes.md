# Conditional density of (y0, y1) given x0..x4

## What the data look like
- No linear correlation between x and y. In the plane the responses form a disc of radius ~4 with
  structure: in polar coordinates the angle is tied to a linear index u = a.x (two bands of slope
  -1, so the angle is u mod pi), and the radius depends on u in a nonlinear way.
- Rotating y by -u showed **two populations**:
  1. a **ring**: a circle centred at 0 with radius R(u) = 3.5 * sigmoid(0.27 - 1.96 u) and a uniform angle
     (I checked the angle histogram, and there is no ellipticity);
  2. a **bar**: points at +-m along the direction at angle u, with isotropic noise, where the bar
     position m = 3.4 * sigmoid(v) depends on a second linear index v = b0 + b.x.
  The ring weight is w = sigmoid(-1.64 u). The ring's radial noise grows with v.
  After fitting, the noise residuals had excess kurtosis of ~0.6, so the noise is a 2-scale Gaussian
  mixture (22% small, 78% larger by a factor of 1.79 in sd), shared by the ring and the bar.

## Program
`Mixture` of 4 parts: ring(s), ring(l*s), bar(sb), bar(l*sb), with weights w*pi, w*(1-pi), ...
The ring (uniform angle) is discretised as a 5x5 nested Mixture of 25 MultivariateNormals placed on the
circle. Each one is stretched along the tangent (covariance s*I + j*t t'), which makes the ring
smooth. The bar is a symmetric 2-Gaussian mixture. 22 fitted parameters, 47 literal constants, 469 AST nodes.

## What I tried (validation NLL, 70/30 split with seed 0, 1500 held-out rows)
| model | val NLL |
|---|---|
| single Gaussian (approx.) | ~4.0 |
| +- mean with rotating direction, 2 Gaussians | 3.30 |
| noisy ellipse, y = R(phi) diag(a, a cos tau) (cos t, sin t) + noise | 3.15 |
| ring + bar, all parameters exp/sigmoid-linear in x (44 params) | 2.233 |
| ring + bar, shared index u for angle/radius/weight (31 params) | 2.230 |
| + bar position m = A*sigmoid(linear) (it saturates) | 2.092 |
| compact version: constant isotropic bar noise, ring noise exp(a + b v) | 2.094 |
| + 2-scale noise mixture (**final form**) | **2.084** |
| + cross-index terms (R and w on v, m on u) | 2.082 (not kept: gain too small) |

The parameters were fit by maximum likelihood with L-BFGS, using exactly the density that the
submitted program computes (same 25-component ring approximation).

## Honest estimate
5-fold CV of the final form (refit for each fold; the warm start comes from an earlier fit on the
70% split): fold test NLLs 2.103, 2.077, 2.171, 2.063. Fold 4 did not finish before its 20-minute timeout, so it is
missing. The mean over the 4 completed folds is 2.104. The fold-to-fold spread is ~0.04, which is consistent with the per-point SE of ~0.045 per
fold. The final model was refit on all 5000 rows; its in-sample NLL is 2.095. The model has few
parameters (22) for 5000 rows, so the optimism is small: the train-to-test gap is about 0.01.

expected_nll: 2.10
