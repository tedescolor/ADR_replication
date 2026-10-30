# Notes

## Data
(y0,y1) is a noisy ring/disk with a hole: angle ~uniform (weak 2nd-harmonic, x-dependent, y0*y1 corr 0.09),
radius = 3-component mixture (means ~1.5, 1.9, 2.7 at x=0) whose means and mixing weights depend linearly on x0..x4.
Gaussian baseline NLL ~4.05. Ideal polar model (r mixture, uniform angle) CV ~3.21 but not expressible exactly with the allowed families.

## Final program
Sequential(y0 marginal, y1|y0):
- radius R_k = b_k + x.a_k (k=1..3), weights softmax(linear in x).
- y0 marginal = mixture over k of "ring marginal" (5 normals: centre + 2 pairs at +-B*R, scale prop. to R; shared shape constants).
- y1|y0 = mixture over k (weights = posterior of k given y0) of two Gaussians at +-sqrt(R^2-y0^2), sd = sigma_k R / sqrt(m^2 + c^2),
  with sign probability sigmoid((t0 + x.t) * y0) to capture the x-dependent y0*y1 association.
Contract: 477 AST nodes, constants within 80 (check passes).

## Validation
5-fold CV (fit by L-BFGS on 4/5 of data, NLL on held out fifth), parameters fitted in numpy replica of model.py (verified equal to model.py NLL).
- single-index radius model, 3 comps: 3.65
- full linear means/weights, 3 comps: 3.360
- + sign term: 3.353
- + 5-component ring marginal (final): 3.344
In-sample NLL of the final fit 3.318. No other model families were compared extensively (time); K=4, x-dependent radial spread and angular modulation untested.

expected_nll: 3.35
