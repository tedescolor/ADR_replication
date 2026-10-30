# Notes

## Data
250 rows, x0 in [0,1]. Scatter shows a single noisy curve for x < ~0.4 that splits
into two branches: an upper branch rising almost linearly to ~2.5 at x=1 and a lower
branch that peaks near x=0.7 and falls to ~-0.9 at x=1. Noise shrinks with x
(sd ~0.35 at x=0 down to ~0.1 at x=1). Within each branch, residuals look Gaussian
(standardized residual skew 0.07, excess kurtosis 0.14).

## Model family
Two-component Normal mixture with a constant weight:
- lower mean m1 = polynomial in x;
- upper mean m2 = m1 + hinge polynomial in h = max(x - c, 0), so both components
  coincide before the split and the weight needs no x dependence;
- a shared log-linear sd: s = exp(a + b x).

## Validation
Maximum likelihood fits (BFGS then Powell, several random restarts), scored with
5-fold CV repeated 2 times (two separate seed sets), plus a paired 4x5-fold run.
CV mean NLL (lower is better):

| model | #params | CV run 1 | CV run 2 |
|---|---|---|---|
| hinge, cubic m1, lin+quad hinge, log-lin sd (hinge321) | 10 | 0.278 | 0.291 |
| hinge421 (quartic m1) | 11 | 0.273 | 0.280 |
| hinge422 (quartic m1, quadratic log-sd) | 12 | 0.273 | - |
| hinge322 | 11 | 0.280 | 0.296 |
| hinge321 + separate upper sd | 11 | 0.282 | |
| hinge321 with Student-t components | 11 | 0.279 | |
| hinge311 (linear hinge) | 9 | 0.331 | |
| constant sd | 9 | 0.432 | |
| m2 = max(m1, separate poly) variants | 9-11 | 0.279-0.295 | |
| sinusoidal lower mean + hinge (sin22) | 12 | | 0.293 |

The log-linear sd matters a lot, and so does the quadratic hinge term. Student-t,
a separate sd per branch, and the sinusoidal mean gave no gain. hinge421 was best
in both CV runs. Differences between the top models are about 0.005-0.01, roughly the
size of the variation between CV seeds.

## Final program
hinge421: 11 fitted constants (plus literals 0.0 and 1.0), refit on all data
(train NLL 0.221). The fitted hinge sits at x≈0.10 with a negative linear term,
so between x≈0.1 and 0.5 the second component sits slightly below the first. That
widens the low-x noise downward. Above x≈0.5 it becomes the rising upper branch.

The CV estimates for hinge421 are 0.273 and 0.280. I add a small allowance for
selection optimism.

expected_nll: 0.285
