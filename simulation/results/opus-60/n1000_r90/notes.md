# Notes

## Data
1000 rows, x0 in (0,1). Scatter shows a single noisy wavy curve for x < ~0.5 that splits into
two branches (one rising to ~2.5, one falling to ~-1) for x > 0.6, with roughly equal
mass on each branch. Noise sd shrinks from ~0.3 at small x to ~0.08 at x=1. Bins at small x
have negative excess kurtosis, which suggests two slightly separated components even there.

## Candidates (5-fold CV NLL, same folds; parameter count k)
- 2-Normal mixture, separate polynomial means (deg 5-7), separate log-sd polys, logistic weight:
  CV 0.226-0.231 (k 18-22). The fitted weight stayed at ~0.5 and both sds were nearly equal.
- Structured: equal-weight mixture of Normal(f(x) +/- h(x), s(x)), with f, h, log s polynomials
  in t = 2x-1:
  - (deg f, h, log s) = (5,3,2) 0.2205; (6,3,2) 0.2213; (6,4,2) 0.2178; (7,4,2) **0.2151**;
    (8,4,2) 0.2172; (7,5,2) 0.2192; (7,6,2) 0.2187; (7,4,3) 0.2150; (8,5,2) 0.2214; (9,6,2) 0.2201
  - separate sds plus a free weight: 0.2229 (worse); Student-t components: 0.2201 (no gain over Normal)
- 10-fold CV with a different seed, for (7,4,2): 0.2155.

## Final
Equal-weight 2-Normal mixture at f +/- h, with deg-7 f, deg-4 h and a quadratic log-sd that both
components share. 16 fitted constants (+ 4 structural literals). The full-data train NLL is 0.1947.
I picked (7,4,2) over (7,4,3) because the CV is the same and it has fewer parameters.
The estimate below adds a little to the CV figure to allow for the optimism of picking the model from the grid.

expected_nll: 0.218
