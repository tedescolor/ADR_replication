# Notes

## Data
250 rows, one covariate x0 in [0,1]. Scatter shows two branches:
- a wavy "lower" branch over the whole range (min ~ -0.85 at x~0.25, max ~ 0.15 at x~0.7, down to -0.9 at x=1);
- a straight "upper" branch y ~ 5x - 2.4 that appears for x above ~0.3 (where it crosses the lower curve)
  and holds about half of the points from there on.
Noise sd falls from ~0.3 at x=0 to ~0.12 at x=1, and looks about the same on both branches.

## Validation
5-fold CV (seed fixed). For each fold the model is refit from a fixed starting point (not warm-started from the full fit),
and I report the mean held-out NLL. One sweep used 2 repeats of 5-fold.

## Candidates (CV NLL)
- 2-normal mixture, poly5 lower mean, linear upper mean, separate log-linear sds, logistic-in-x weight: 0.239 (train 0.175)
- same but lower mean = a*sin(2*pi*f*x+phi)+c+d*x: 0.228; shared sd: 0.219
- weight = cap * sigmoid(k(x-c)) with free k, c (k went to a hard step): shared sd 0.193, separate sds 0.213
- fixing the step location c (k=300): the profile is flat for c in 0.2..0.5 (CV 0.173-0.183) because the line sits close to the
  lower curve there. Using a smooth step c=0.33, k=20 gives 0.178, which hedges on where the branch starts.
- Student-t noise: df went to infinity, so normal noise is enough. Constant sd: 0.269 (much worse). Quadratic log-sd: 0.182 (no gain).
- **cap fixed at 0.5** (the fitted cap was 0.47-0.48): **CV 0.174**, train 0.152. Chosen.

## Final program
Mixture of 2 normals with a shared sd: s = exp(-1.05 - 1.078 x).
- lower: 0.819 sin(5.4224 x - 2.436) + 0.283 - 1.34 x
- upper: 5.033 x - 2.388
- upper weight w = 0.5 * sigmoid(20 (x - 0.33))
That is 13 constants. I picked it because it had the best CV and the fewest free parameters.

## Expected NLL
CV gives 0.174. I chose this model after comparing about 15 variants, so I add a small allowance for selection optimism.

expected_nll: 0.18
