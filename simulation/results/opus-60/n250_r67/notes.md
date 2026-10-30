# Notes

## Data
250 rows, x0 in [0, 1], y in [-1.36, 2.61]. Binned summaries show:
- for x < ~0.5 there is a single smooth branch (dips to about -0.77 at x≈0.2, rises to about 0 at x≈0.5),
  with noise sd shrinking from ~0.38 to ~0.24;
- from x ≈ 0.55 the data split into two branches: an upper one that rises roughly linearly to about 2.5
  at x=1, and a lower one that goes back down to about -1.1 at x=1. The two are roughly equally populated,
  and their noise sd is small (~0.1–0.15).

## Models tried
A two-component Gaussian mixture with polynomial parameters:
lower mean (degree dm), lower log-sd (ds), upper mean (du), upper log-sd (dsu), and the logit of the upper
weight (dw). I initialised from a crude split of the points into branches, then ran BFGS from several
perturbed starts.
- A first unconstrained mixture fit (random init, quadratic log-sds) reached a train NLL of 0.09, but one
  component's sd collapsed to near 0 at the edges. Its 5-fold CV was about 0.28 and unstable, so I dropped it.
- Structured grid, scored with 5-fold CV repeated 4 times. Differences are paired per point and the
  standard errors below are for those paired differences:
  - (3,2,2,1,1): 0.285
  - (3,2,2,1,2): 0.255
  - (3,2,1,1,2): 0.249
  - **(3,1,1,1,2): 0.241**
  - (3,1,1,0,2): 0.2415
  - (4,1,1,1,2): 0.2417
  - (3,1,2,1,2): 0.242
  - (3,1,1,1,3): 0.248
  - (5,2,2,1,2): 0.250
  - (3,0,1,1,2): 0.331, so a heteroscedastic lower branch is clearly needed
  - (2,2,2,1,1): 0.66, so a cubic lower mean is clearly needed
  - The weight needs a quadratic logit: a linear one is worse by 0.03 ± 0.014.
  - Most of the other differences are within about 1 SE of each other.

## Final program
`Mixture([1-p, p], [Normal(cubic(x), exp(linear(x))), Normal(linear(x), exp(linear(x)))])`
with `p = sigmoid(quadratic(x))`. It has 16 constants (15 fitted parameters plus a 1.0), and I refit it on
all 250 rows using 20 starts. Train NLL is 0.178 and repeated 5-fold CV NLL is 0.241. I chose it because it
is the simplest model at the top of the CV ranking.

The fit is only validated for x in [0, 1], which is the observed range. Outside that range the polynomials
extrapolate.

## Estimate
CV models were fitted on 80% of the data, so the full-data fit should be slightly better than 0.241.
To allow for selection optimism, I keep the estimate near the CV value.

expected_nll: 0.24
