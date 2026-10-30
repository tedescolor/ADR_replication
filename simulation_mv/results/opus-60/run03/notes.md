# Notes

## Data exploration
- y = (y0, y1) lies in a disk of radius ~4. Its radius r = |y| has a sharp ring at r ~ 3.1 plus
  an inner spread (r ~ 0.5-2.5); the angle is roughly uniform marginally, with no first-harmonic
  dependence on x (the law is symmetric under y -> -y).
- The radius is strongly tied to two linear covariate indices:
  u ~ (0.92, -0.63, 0.46, -0.33, 0.69).x and v ~ (-0.27, 0.57, -0.16, 0.37, 0.20).x. A scatter
  of r against u shows a "Y": one branch with r ~ |1.93 + u|, one with r ~ 1.99 + v, and an outer
  ring at r ~ 3.0 + 0.14 u.
- Second-harmonic angular structure (cos 2θ, sin 2θ) depends on x: the "v" branch is
  concentrated at two opposite points ±(1.99 + v)(cos φ, sin φ) with φ ≈ π - 0.79 u (two
  spiral-arm-like blobs).

## Model (final)
A 3-component mixture with softmax weights that are linear in (u, v):
1. ring of radius |c + u| (uniform angle, radial sd 0.32);
2. an axial pair of rotated Gaussians at ±(c + v)(cos φ, sin φ), φ linear in u (radial sd
   0.34, tangential sd 0.24), written as a Mixture of two MultivariateNormal;
3. outer ring of radius 3.0 + 0.14 u (sd 0.20).
A noisy uniform ring is written as Sequential(y0 marginal = 3-normal symmetric mixture scaled
by the radius, y1 | y0 = two normals at ±sqrt(R² - y0²) whose sd is inflated near the
tangent). Its shape constants are shared by both rings and fitted by maximum likelihood together
with everything else. 42 constants, 487 AST nodes (the 500-node limit was the binding
constraint). To fit under it I dropped the four index terms whose removal cost <0.002 nats on the
holdout (measured one by one, without refit). Refitting with those terms constrained to zero
landed in a worse optimum (2.385), so the unconstrained fit is used with those terms omitted.

## Validation
One fixed random 75/25 split (seed 1); every candidate was fitted on the 75% part by
L-BFGS maximum likelihood, and the generated program was scored on the 25% part.
| model | test NLL |
|---|---|
| 2-comp radial mixture, uniform angle (r-only analysis) | 3.69 |
| 3 rings, radius linear in x (program) | 3.262 |
| + axial von Mises modulation of the rings | 2.885 |
| ring / Gaussian pair / ring, free linear functions (too many AST nodes) | 2.354 |
| same with two shared indices u, v (all terms) | 2.330 |
| same, abs(coef)<0.06 terms dropped (incl. two that matter) | 2.351 |
| **final**: only the 4 harmless terms dropped, pair variances precomputed | **2.331** |
Train and test NLL agree closely (2.325 vs 2.330), so overfitting is small with about 40
parameters on 3750 rows. The final constants were refitted on all 5000 rows.

expected_nll: 2.335
