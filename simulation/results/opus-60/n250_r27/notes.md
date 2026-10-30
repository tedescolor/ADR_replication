# Notes

## Data
250 rows, x0 in (0,1), y in (-1.45, 2.57). Scatter shows two noisy curves present over the
whole x range: an "upper" curve (about -0.4 at x=0.2, rising to 2.5 at x=1) and a "lower"
wave-shaped curve (min about -1.1 near x=0.2, max about 0.1 near x=0.7, -1 at x=1). They nearly
overlap for x<0.5, which is why the residuals there looked flat-topped (negative kurtosis) when
I fit a single curve. Noise shrinks with x (sd about 0.3 at x=0 down to about 0.12 at x=1).

## Validation
10-fold CV repeated 3 times (different permutations), mean held-out NLL per row. Each fold is
refit by BFGS, warm-started from the full-data optimum (this avoids label-switching and local
optima; it may make CV slightly optimistic).

## Candidates (CV NLL)
- 1 main curve plus 1 linear "branch" component with logistic weight (first idea): 0.293 – 0.37
- 2 polynomial curves, Normal noise, log-linear sd:
  - cubic/quintic, separate sd: 0.280; shared sd: 0.268
  - cubic/quartic, shared sd: 0.261; separate sd: 0.271; quadratic log-sd: 0.273
  - cubic/cubic: 0.269; quartic/quartic: 0.264; quadratic/quintic: 0.319
  - constant sd: 0.304
  - sinusoid lower curve (with/without trend): 0.275 / 0.277
  - logistic-in-x weight: 0.274 (no gain)
  - Student-t noise: 0.261 (no gain over Normal)
  - **cubic/quartic, shared log-linear sd, weights fixed at 0.5: 0.257 (best)**

## Final program
Mixture of 2 Normals, weights 0.5/0.5, means = cubic (upper) and quartic (lower) polynomials in
x0, shared sd = exp(a + b*x0). 11 fitted parameters (13 literals incl. the two 0.5 weights).
Train NLL 0.208, CV NLL 0.257. The final fit uses all 250 rows, slightly more than each CV
fold, but the warm-started CV may be a bit optimistic, so I estimate about 0.26.

expected_nll: 0.26
