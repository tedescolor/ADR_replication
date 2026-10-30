# Notes

Data: 1000 rows, x0 in (0,1), y in (-1.7, 2.6). Scatter shows a noisy single band for
x < ~0.5 (sd ~0.3) that splits into two tight branches for x > 0.55 (upper rising almost
linearly to ~2.4, lower bending down to ~-1; sd ~0.1-0.2). Noise shrinks with x.

Model family: 2-component Gaussian mixture, means polynomial in x, log-sd polynomial in x,
mixture weight logistic in x. Fitted by maximum likelihood (BFGS, several inits).

Tried (CV NLL, 5-fold x2 with re-fit from fixed inits, later 10-fold x2/x3 warm-started):
- "trunk + branch" structure (lower curve poly4 over whole range, upper branch poly1-2
  switched on by a steep sigmoid): ~0.230; with skew-normal trunk and quadratic log-sd 0.227.
- free 2-component mixture, both means cubic, both comps present everywhere
  (the components cross inside the trunk, making the trunk noise slightly platykurtic,
  which matched the negative excess kurtosis of trunk residuals): 0.219-0.223.
  Means cubic/cubic, log-sd quadratic/linear, weight logit linear: CV 0.2204.
- Same but weight logit QUADRATIC in x: CV 0.2133 (10-fold x2, warm-started), in-sample
  0.1994. This clear gain (~0.007) made it the final model (16 fitted constants + the 1.0).
  Cubic weight logit (0.2146) or quartic m2 (0.2140) did not help further.
  In this fit comp 1 has almost zero weight near x=0 (weight ~0.001), so its tiny sd there
  is harmless.
- Higher degrees (quartic means, quadratic sd for comp 2, cubic log-sd) gave changes of
  ~0.001, well inside the noise (SE of the mean NLL ~0.023, paired differences much smaller),
  so I kept the simpler program.

Validation: K-fold CV on data.csv (refit on each training fold). Final model: in-sample
NLL 0.1994, CV NLL 0.2133 (10-fold x2; SE across points ~0.023). Expected held-out NLL taken as the CV value plus a small
allowance for model selection optimism.

expected_nll: 0.217
