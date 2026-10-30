# Notes

## Data
1000 rows, x0 in (0, 1), y in about (-2.1, 2.7). The scatter plot shows one noisy curve for x0 < ~0.5.
Above that it splits into two branches: an upper one that rises roughly linearly to ~2.5, and a lower one
that peaks near 0.2 and falls to ~-1. The noise shrinks as x0 grows, from sd ~0.33 at x0=0 to ~0.10 at x0=1.

## Validation
I used 5-fold CV, repeated with 2 different shuffles for the final comparisons. For each fold I fit by
maximum likelihood (BFGS, a few restarts) and score the held-out mean NLL.

## Candidates (CV mean NLL)
- 2-component Gaussian mixture with free polynomial means (in t = 2*x0-1), separate log-linear sds, weight 0.5:
  deg3 0.207, deg4 0.192, deg5 0.196. Using quadratic log-sd or a free constant weight did not help.
- A structured "split" model (a shared curve plus softplus divergence after a breakpoint) scored 0.197–0.202,
  which is worse. The free mixture prefers two curves that cross near x0≈0.48, so the left region is
  also best explained as two overlapping branches.
- Free quartic means with a *shared* sd (the per-branch sds were nearly identical in the residual analysis):
  - log-linear sd: 0.1878
  - log-quadratic sd: 0.1882
  - sd linear in x: **0.1858** (best)
  - Student-t version: 0.1867
  - degrees (3,4), (5,4) and (5,5): 0.1874, 0.1864 and 0.1877
- Final choice: equal-weight mixture of two Normals, quartic means, shared sd = a + b*t (floored at 0.03
  for safety outside [0,1]). This uses 17 literals in total. Train NLL is 0.1720 and CV NLL is 0.1858.

The CV differences between the top few variants are around 0.001–0.002, about the size of the noise.
I picked the simplest of the best ones. The CV estimate is nudged up slightly for selection optimism.

expected_nll: 0.188
