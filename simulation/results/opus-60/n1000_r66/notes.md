# Notes

## Data
One covariate x0 in (0,1), y continuous. Scatter shows a single noisy "trunk" for
x < ~0.5 (U-shaped, min ~ -0.85 at x~0.2) that forks at x ~ 0.5 into an upper branch
(rising to ~2.4 at x=1) and a lower branch (falling to ~-1 at x=1), roughly 50/50.
Residual sd decreases steadily with x (~0.38 at x=0 to ~0.10 at x=1); residuals are
close to Gaussian.

## Validation
5-fold CV repeated with 2 random splits (10 fits per spec), refitting all parameters
by maximum likelihood (BFGS, 3 restarts) on each training fold; mean held-out NLL reported.

## Candidates (CV NLL)
- 2-normal mixture, independent quartic means, log-linear sds: 0.170 (train 0.150).
  Quintic: 0.171; quadratic log-sd: 0.175.
- Fork model: upper mean f(x) polynomial, lower mean f(x) - softplus-ramp(x-c)*(linear);
  the ramp width collapsed to ~0, i.e. a hard hinge at c ~ 0.50.
  - quartic f, separate log-linear sds: 0.1664
  - quintic f: 0.1694; quadratic D: 0.1676; quadratic log-sd: 0.1678
  - Student-t components (nu ~ 34, basically normal): 0.1663
  - quartic f, shared sd: 0.1671
  - **cubic f, shared log-linear sd: 0.1665** (12 params)

## Final
y ~ w N(f, s) + (1-w) N(f - d, s*e^0.02), with t = 2x-1,
f = cubic in t, d = max(x-0.505, 0)*(3.696+3.327 t), s = exp(-1.558-0.647 t),
w ~ 0.5. 12 constants. I picked it as the simplest model that is within CV noise
of the best one. Train NLL is 0.1487 and CV NLL is 0.1665. My estimate for new data
is the CV value, rounded up a little.

expected_nll: 0.167
