# Notes

## Data
250 rows, x0 in [0,1]. Scatter shows a single noisy curve for x < ~0.55 (dips to about -0.8
near x=0.25, noise sd about 0.4 near x=0 shrinking to about 0.2), then a split into two roughly
equally populated branches: an upper one rising to about 2.5 at x=1 and a lower one falling
to about -0.8.

## Model (final)
A two-component Gaussian mixture:
- common cubic trunk m(x);
- split variable s = softplus(k (x - t0)) / k with t0 = 0.546, k = 20.6;
- branch means m + a1 s + a2 s^2 (upper) and m + b1 s + b2 s^2 (lower);
- shared log-linear sd, exp(c0 + c1 x);
- weight 0.5 on each branch for x > t0, otherwise a single Normal (hard switch). The fitted
  logistic weight steepness went to about 5000 and a free post-split weight came out at 0.505,
  so I fixed both.
12 fitted parameters, 18 literals. Fitted by ML (Nelder-Mead followed by BFGS, with restarts).

## Validation
5-fold CV, repeated with different random partitions, refitting every candidate on each
training split.
- base (cubic trunk, log-linear sd, weight 0.5): CV NLL 0.160 to 0.169 (SE about 0.023); train NLL 0.110
- log-quadratic or log-cubic sd: CV 0.170 to 0.175 (worse)
- quartic trunk: 0.171 or worse
- logistic weight in s: 0.18 to 0.20 (worse)
- branch-specific sd: 0.180 (worse)
- Student-t components (nu about 7): paired CV 0.170 against 0.160 for Normal, a difference of
  0.0099 with SE 0.005, so the t version was rejected.
In an earlier run (variable-softness switch, polynomial degree 3 to 6) the CV grew with
degree: 0.185, 0.189, 0.192, 0.205.

I chose the simplest model, which also had the best CV. I add a small allowance for
selection optimism to the CV mean.

expected_nll: 0.17
