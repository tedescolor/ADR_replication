# Notes

## Data
250 rows, x0 in [0,1]. y vs x0 looks like one noisy wavy curve (min ~-0.8 near x=0.25,
rises to ~0.3 at x=0.6). Around x=0.4-0.6 it splits into two branches: an upper one that
climbs fast to ~2.5 at x=1 (about 55-58% of the points), and a lower one that keeps the
wave shape and drops to ~-1 at x=1. Noise sd falls with x (~0.35 at x=0, ~0.1 at x=1).
Residuals around the branches look Gaussian (skew ~0, excess kurtosis ~0).

## Model family
A two-component Normal mixture with constant weights:
- lower mean m(x) = smooth curve; upper mean = m(x) + hinge(x - t);
- log-sd of each component linear in x.
Below the split t both components share a mean, so the density there is a scale mixture.

## Validation
5-fold CV, repeated over 2 random partitions. Each training fold refits everything,
including a grid profile over t. CV NLL (mean per held-out row):
- sinusoid lower + linear hinge, NM fit, 1 partition: 0.238 (0.247 over 2 partitions with profiled t)
- sinusoid + quadratic hinge: 0.253; with quadratic log-sd: 0.249 / 0.271
- softplus (smooth) hinge with sinusoid: 0.239 (1 partition)
- poly lower deg 3/4/5, linear/quadratic hinge, lower-branch hinge, quadratic log-sd:
  0.297, 0.253, **0.236 (deg4 + linear+quadratic hinge, linear log-sd)**, 0.242, 0.256, 0.252, 0.272, 0.244
Extra flexibility (lower hinge, quadratic log-sd) consistently hurt out of sample.

## Final
Quartic lower mean in (x-0.5); upper = lower + 0.039 h + 9.33 h^2 with h = max(0, x-0.38);
log-sd linear per component; weights 0.449/0.551. 13 fitted parameters (17 literals in model.py).
Train NLL 0.180, CV NLL 0.236. Folds vary by roughly ±0.01 and the model was chosen as the
best of several, so I shade the estimate up a little.

expected_nll: 0.245
