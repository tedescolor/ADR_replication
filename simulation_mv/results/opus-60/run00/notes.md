# Notes

## Exploration
- y is point-symmetric (y -> -y). In polar coordinates: a radial variable plus an angle.
- Rotating y by an angle alpha(x) linear in x reveals two parts:
  (1) a "bar": two symmetric blobs at +-mb(x) along the rotated axis, thin across it;
  (2) an isotropic "ring" of radius R(x), uniform angle.
- The gate between bar and ring is a logistic in x (coefficients roughly proportional to alpha's).
- mb(x) and R(x) depend on different linear combinations of x and saturate (S-shaped).

## Models (validation: 5-fold / 80-20 holdout, ML fit with L-BFGS in numpy, then
re-checked through adr on the same data)
- single MVN: ~4.0
- M1: bar (2 rotated MVNs) + ring (Sequential: 3-normal y0 marginal, y1|y0 = +-sqrt(R^2-y0^2)),
  linear mb, R: 5-fold CV NLL 2.535 (train 2.535).
- Diagnostics of M1 showed the bar half-length and ring radius are S-shaped in their linear
  predictors and the true radial noise is small (~0.2), so M1's linear radii inflated sigma.
- M2: mb = b0 + b1 tanh(lin_b(x)), R = c0 + c1 tanh(lin_c(x)); ring y0 marginal as a
  5-normal symmetric mixture (scales sqrt(a R^2 + b)); 80/20 holdout NLL 2.0966 (train 2.0657).
- M3 (final): same as M2 with scales a R + b (needed to fit the 500-AST-node limit);
  holdout NLL 2.0969 (train 2.0658). Its adr evaluation matches the numpy code.

## Final program
Mixture(w(x), [bar, ring]):
- alpha(x) linear (6), gate w = sigmoid(linear) (6), mb and R tanh-saturating (8 each);
- bar = 50/50 mixture of two MVNs at +-mb(cos a, sin a), covariance rotated diag(su^2, sv^2);
- ring = Sequential(5-normal y0 marginal scaled by R, y1|y0 = +-sqrt(R^2-y0^2) with
  slope-adjusted noise). 53 constants, 499 AST nodes.
Fitted by maximum likelihood (L-BFGS) with the parameter vector shared across all pieces.
The final constants are from the full-data refit (train NLL 2.0708, same model form
as the version validated on the 80/20 split).

The expected NLL is the 20% holdout value plus a small allowance for split noise
(SE of the holdout mean is ~0.03).

expected_nll: 2.10
