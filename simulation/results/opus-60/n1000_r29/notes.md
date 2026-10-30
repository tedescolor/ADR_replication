# Notes

## Data
1000 rows, x0 in (0,1). For x0 < ~0.55, y follows one noisy curve (it dips to about -0.8 near x=0.25),
with sd around 0.3. Past ~0.6 it splits into two roughly equal branches: an upper one rising to
about 2.6 at x=1 and a lower one falling to about -1.0. The noise shrinks as x grows (sd about 0.09 at x=1).

## Validation
5-fold CV (permutation seed 1), mean held-out NLL per row. I checked the top candidates on two more
fold permutations (seeds 2 and 3). All fits are direct maximum likelihood (BFGS, several random starts),
using a Chebyshev basis while fitting; the final model converts it to power form.

## Candidates (CV NLL, seed 1)
- 2-normal mixture, polynomial means of degree d, shared log-sd polynomial of degree k, constant weight:
  d=3: 0.213, d=4: 0.196, **d=5,k=2: 0.1906**, d=6: 0.194, d=7,k=3: 0.191, k=1: 0.202
- separate component scales: 0.194; weight depending on x: 0.193-0.200; Student-t components: 0.189-0.190
  (not a robust gain)
- structured "shared curve + softplus branch split" (one normal for low x, two branches after a
  fitted breakpoint): 0.194-0.197. A single normal is about as good as two components for x<0.5,
  but overall this model did not beat the free mixture.
- sqrt(x) basis term: 0.197
- skew-normal components: 0.1883 on seed 1, but 0.2057 on seeds 2-3 against 0.1972 for the baseline,
  so it is unstable and I rejected it.

## Final
Mixture of 2 normals with quintic means, a shared scale exp(quadratic), and a constant weight
(about 0.52/0.48). 17 constants. Refit on all data: training NLL 0.1743. CV NLL: 0.1906 (seed 1),
0.1972 (seeds 2-3), about 0.195 overall.

expected_nll: 0.196
