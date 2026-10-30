# Notes

## Data
726 rows, 6 covariates. x3, x4 are jittered small integer counts (mostly 0). y has a slightly right-skewed distribution.
An OLS linear mean leaves heavy-tailed residuals (excess kurtosis 3.3, skew 0.5). The residual scale clearly depends
on the covariates: it is larger at both extremes of x5 and at high x0, and it rises with x1.

## Validation
10-fold CV of the mean held-out NLL, with all parameters refit by maximum likelihood (L-BFGS) in each fold.
Model selection used seed 0. The final candidates were re-scored on different fold splits (seeds 11, 12, 13; 30 fits each).

## What was tried (CV NLL)
- Gaussian, homoscedastic, linear mean: 1.397
- Gaussian, log-scale linear + x0^2, x5^2: 1.284
- Student-t, homoscedastic: 1.331; t, log-scale linear + quadratics: ~1.257 (first saved baseline)
- Skew-normal: 1.265. Laplace: 1.39. 2-component normal mixture noise: 1.251 (small gain over t on the baseline).
- Adding squares or interactions to an OLS mean on their own did not help: the outliers dominate.
- Greedy forward selection over mean and log-scale terms (squares, pairwise products, |x5-c|) with t noise
  picked the mean terms x0*x5, x3*x5, x2*x4, x1^2 and the scale terms |x5-c|, x0*x5, x1*x3.
- Re-scored on fresh splits: base 1.260, greedy set 1.207, **greedy mean + full linear log-scale + |x5|, x0x5, x1x3: 1.191**,
  and the same plus mixture noise: 1.202 (worse, so rejected).

## Final program
Student-t with a fitted df (nu = e^1.644, about 5.2). The mean is linear in all 6 covariates (centred) plus x0*x5, x3*x5, x2*x4 and x1^2.
The log-scale is linear in all 6 plus |x5-2.118|, x0*x5 and x1*x3, clipped to [-2, 1] so the scale stays sane at the corners of
the covariate box. This clipping does not change the in-sample fit. 30 constants, 276 AST nodes. In-sample NLL 1.154; CV NLL 1.191.
The terms were chosen by CV on this same data, so the CV figure is somewhat optimistic. I add about 0.02 for that.

expected_nll: 1.21
