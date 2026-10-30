# Notes

## Data
800 rows, 6 covariates (x0 count 1..51, x1 0..1865 heavy right tail, x2 count 0..83, x3 binary,
x4 in (0,25), x5 heavy right tail), y0 and y1 both positive and right-skewed, corr(y0,y1)=0.55.
Features used: f0=log x0, f1=log(1+x1), f2=log(1+x2), f3=x3, f4=x4/10, f5=log x5.

## Validation
5-fold CV repeated 2x (3x for the final model), with different seeds. All parameters fitted by
maximum likelihood (L-BFGS) on the training folds only; mean held-out NLL (natural log, joint density
in the original y scale, with Jacobians where y was transformed) is reported.

## Candidates (CV NLL)
| model | #par | CV NLL |
|---|---|---|
| log y0 ~ Normal(lin, exp(lin)); y1 given y0 ~ Normal(lin + b log y0, exp(lin)) | 29 | 2.545 |
| same, but log y1 normal | 29 | 2.828 |
| same, but y1 given y0 Gamma | 29 | 2.598 |
| same, but sqrt y1 normal | 29 | 2.529 |
| + f4^2, f5^2 in mean of log y0 | 31 | 2.517 |
| + also f0^2,f1^2,f2^2 | 34 | 2.515 |
| + (log y0)^2 in mean of y1 | 32 | 2.501 |
| + log y0 in log-scale of y1 | 33 | 2.494 |
| + quadratics in mean of y1 too | 35 | 2.496 |
| sqrt-normal y1 instead | 33 | 2.502 |
| SkewNormal for y1 given y0 | 34 | 2.487 |
| + skew-normal for log y0 | 35 | 2.488 (no gain) |
| y0 raw-scale skew-normal | 35 | 2.511 |
| + cubic (log y0) / extra scale terms / quadratic scale of y0 | 36-37 | 2.489-2.492 (no gain) |
| **SkewNormal y1 + indicator x0==1 in mean of log y0** | 35 | **2.480** (3x5-fold) |

## Final program
Sequential(LogNormal(mu0, s0), y1 given y0 ~ SkewNormal):
- mu0 = linear in f0..f5 + f4^2 + f5^2 + [x0==1] indicator; log s0 = linear in f0..f5;
- y1 given y0: loc = linear in f0..f5 + a*log y0 + b*(log y0)^2, log scale = linear in f0..f5 + c*log y0,
  constant shape 1.48.
35 fitted constants (about 40 literals in total, including feature constants), fitted on all 800 rows.
In-sample NLL 2.428, CV 2.480. Diagnostics showed curvature in x4 (U-shaped) and in y1 against log y0,
plus right skew in the y1 residual; those were what each added term fixed. Additions that didn't
improve CV were left out.
Valid over the whole covariate box (all features are log/linear with x0>=1, x5>0, x1,x2>=0),
and the contract check passes.

Expected NLL on new data: CV estimate 2.480. Refitting on all data should help a little, while
selecting the model on the same CV pulls the other way a little, so:

expected_nll: 2.48
