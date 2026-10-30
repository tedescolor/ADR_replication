# Notes

## Data
1000 rows, x0 in (0,1). Scatter: a single noisy trunk (sd ~0.3) for x < ~0.6 following a
wavy curve (0.4 at x=0, min -0.8 near x=0.2, rising to ~0.2 at 0.6), which then splits
into two branches with roughly equal weight: an upper, nearly linear branch rising to ~2.5
at x=1, and a lower branch continuing the trunk curve down to ~-0.9. Noise shrinks with x
(sd ~0.25 -> ~0.09).

## Validation
5-fold CV (fixed permutation) of the mean NLL, maximum likelihood fits (BFGS, 8 restarts).
(Note: the very first comparison batch was accidentally run as 4 parallel processes; all
later runs used at most 2.)

| model | #params | train | CV |
|---|---|---|---|
| free 2-comp normal mixture, poly means deg3, log-sd deg1, const w | 13 | 0.222 | 0.241 |
| same, means deg4, log-sd deg2 | 17 | 0.200 | 0.226 |
| same, means deg5/sd2/w1 | 21 | 0.198 | 0.229 |
| same, means deg6/sd2/w2 | 25 | 0.194 | 0.230 |
| **A: trunk poly deg5 f(x); upper = f + b*s*softplus((x-c)/s); shared quad log-sd; const w** | 13 | 0.211 | **0.2249** |
| A with trunk deg4 / deg7 | 12 / 15 | | 0.2272 / 0.2256 |
| B: A with separate linear log-sd per component | 14 | 0.214 | 0.230 |
| C: A + quadratic term in the split | 14 | 0.211 | 0.226 |
| D: A + weight linear in x | 14 | 0.211 | 0.2257 |
| E: A + sd offset for the upper branch after split | 14 | 0.211 | 0.228 |

PIT residual diagnostics of A by x-decile: means ~0, sds 0.86-1.13, no systematic
skew/kurtosis -> Gaussian components are adequate.

## Final
Model A: 2-component Normal mixture. Lower component mean = degree-5 polynomial in
t=2x-1 (the trunk + lower branch); upper component mean = same + smooth hinge
(softplus, knot c=0.68, width 0.116); both share sd = exp(quadratic in t); constant
weight 0.52. 13 fitted constants (+ a few structural ones), chosen as the simplest
model with best CV.

CV NLL 0.225; train 0.211. Slight allowance for selection optimism:

expected_nll: 0.228
