# Notes

Data: y vs x0 in [0,1]. For x < ~0.5 there is one noisy band (sd ~0.3); above ~0.55 it
splits into two thin branches: a rising straight line (~ -2.05 + 4.58 x) and a lower
curve that peaks near x=0.7 and drops to -1 at x=1. The noise shrinks as x grows.

Model: a 2-component Gaussian mixture.
- Comp 1 (line): linear mean, constant sd.
- Comp 2 (curve): quartic mean, log-sd linear in x.
- Weight of comp 1 = sigmoid(linear in x). It rises from ~0.03 at x=0 to ~0.6 at x=1.

That is 12 fitted constants, fitted by maximum likelihood (BFGS, a few restarts).

Validation: repeated 5-fold CV. The mixture is refit on each training fold and test NLL
is averaged over all held-out points.
- Polynomial degree of the curve mean: quartic (5 coeffs) CV 0.27-0.30, cubic 0.32,
  quintic and higher 0.30-0.31.
- Constant weight (no x dependence): 0.337.
- Constant sd for the curve: 0.329.
- x-varying sd for the line: 0.275 (no gain).
- Student-t curve component: 0.274 (no gain).
- 3-coefficient weight: 0.2955 (vs 0.2996 for the same seeds). The difference is within
  noise, so I kept the simpler one.
- Sinusoidal curve mean (a + b x + A sin(wx+phi)): 0.30-0.35, no better.

CV estimates for the chosen config were 0.272 (2 repeats, 2 restarts) and 0.300
(4 repeats, 3 restarts). The spread is mostly from fold seeds and local optima. The
final model was fit on all 250 rows, which should help slightly.

expected_nll: 0.29
