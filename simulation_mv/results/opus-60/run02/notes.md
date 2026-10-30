# Notes

## Exploration
- x0..x4 ~ iid N(0,1); y has no linear correlation with x. Scatter of (y0, y1) shows a disc/ring of
  radius up to ~4.
- In polar coordinates (r, theta) the structure is clear. Fitting a circular regression of the
  doubled angle 2*theta on x shows that a large share of points lie on a **line through the origin
  at angle z = a.x** (both directions, theta = z or z + pi). a ~ (-0.72, 0.48, -0.36, 0.24, -0.54),
  intercept ~0.
- The other points have **uniform angle** and lie on a **thin ring whose radius g decreases with z**
  (~3.5 for z << 0, ~0.6 for z >> 0, tanh shaped).
- The fraction of line points is sigmoid(1.68 z).
- Along the line, the distance |t| from the origin is h = 1.98 + 1.20 tanh(b.x) with a second
  projection b ~ (-0.40, 0.87, -0.24, 0.55, 0.31); signs +/- are equally likely.
- Noise around the line points is isotropic Gaussian with sd ~0.23; the ring's radial sd is ~0.2-0.3.

## Models compared (80/20 split, seed 1; NLL on the 1000 held-out rows)
| model | val NLL |
|---|---|
| 5 radius/angle Gaussian-mixture w/ fixed ring blobs, weights linear in x (first try) | 2.835 |
| line + ring, all link functions linear in all 5 x (35 params) | 2.080 |
| line + ring, weight and ring radius depend on z only, isotropic line noise (23 params) | **2.079** |
| + quadratic tanh terms in h and g | 2.079 (coefficients -> 0) |
| **+ line noise as 2-scale isotropic mixture (sd 0.135 / 0.263, 66% wide) — final** | **2.0625** |

The ring (uniform angle, radial Gaussian) is not directly expressible with adr.dist, so it is
approximated by 25 Gaussian blobs equally spaced on the circle (5 x 5 nested Mixtures), radial sd
sr, tangential sd ~0.126 g. A simulation showed this costs < 0.01 nats on ring points vs the exact
ring density.

Fitting: direct maximum likelihood (L-BFGS-B) of the full mixture density, implemented in numpy;
model.py was generated from the fitted parameters and re-evaluated through adr.dist to check it
reproduces the numpy NLL (2.0794 on the val split, identical).

## Validation
- 80/20 split (seed 1): base model 2.0794, final (2-scale line noise) 2.0625.
- 5-fold CV (seed 7) of the base model: 2.065, 2.124, 2.102, 2.048, 1.973 -> mean 2.0626 (se ~0.026);
  train NLL ~0.015 lower than held-out, so overfitting is small (23-25 parameters on 5000 rows).
- Final model on folds 0 and 1: 2.048 / 2.114 vs base 2.065 / 2.124 (gain ~0.014, consistent with the split).
- model.py was checked through adr.dist: NLL on the full data 2.0439 (equal to the numpy fit).
- A 2-scale radial noise for the ring was not tried to completion (too slow, and the 500-AST-node
  budget is used up: final program has 493 nodes).

## Final program
Mixture([w(1-o), w(1-o), w o, w o, 2(1-w)],
        [N(+h u(z), sl^2 I), N(-h u(z), sl^2 I), N(+h u(z), so^2 I), N(-h u(z), so^2 I), ring(g(z))]),
z = a.x (rotation angle, u(z) = (cos z, sin z)), w = sigmoid(c0 + c1 z), h = h0 + h1 tanh(b.x),
g = g0 + g1 tanh(g2 z + g3), ring radial sd exp(r0 + r1 g), tangential sd k g.
Refit on all 5000 rows (intercepts of z and of h's tanh dropped: ~0, and needed to fit the node budget).
44 counted constants, 493 AST nodes.

Expected held-out NLL: CV mean of base model 2.063 minus ~0.014 gain from the noise mixture,
plus a small allowance for selection.

expected_nll: 2.05
