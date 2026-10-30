# Notes

## Data
250 rows, x0 in (0,1). Scatter: a single noisy "trunk" for x < ~0.45 (y≈0 at x=0, minimum
≈-0.78 near x=0.25, back to ≈0 at x≈0.5), which then splits into two branches: an upper one
rising almost linearly to ≈2.5 at x=1, and a lower one bending down to ≈-0.95 at x=1.
Roughly half of the points fall on each branch, independent of x. The noise sd shrinks with
x (≈0.35 near 0, ≈0.17 mid, ≈0.10 near 1). Standardised residuals are close to Gaussian
(skew 0.08, excess kurtosis -0.06).

## Model family
A two-component Gaussian mixture with constant weight. Both components share a polynomial trunk
mean tr(x); after a learned split point c, each branch adds its own a_k*h + b_k*h^2 with
h = max(x - c, 0). Below c the two components coincide, so the density there is unimodal.
There is one log-linear sd shared by both components. Fitted by maximum likelihood
(Nelder-Mead followed by BFGS, with several restarts).

## Validation
5-fold CV repeated twice (10 folds in total), with every fold refitted. Mean held-out NLL (± s.e.):
- cubic trunk, quadratic hinges, shared log-linear sd: 0.210 (0.035)
- the same with separate sd per component (A): 0.228; with a separate sd for the lower branch: 0.218
- log-quadratic sd: 0.210; quartic trunk: 0.210; Student-t components: 0.209 (df≈8 → Normal is fine)
- linear hinge only: 0.230; cubic hinges: 0.217; trunk deg 5 + sd quad: 0.225
- **quadratic trunk, quadratic hinges, shared log-linear sd: 0.2045 (0.034)**, train 0.164

The differences are well within one s.e. I picked the simplest model that tied for the best
CV score: 12 constants (including the 0.0 and 1.0 literals).

## Final
`model.py` uses that quadratic-trunk mixture. In-sample NLL is 0.164 and CV NLL is 0.205. Fitting on
all data should help slightly, but held-out sets vary, so my estimate is about 0.21.

expected_nll: 0.21
