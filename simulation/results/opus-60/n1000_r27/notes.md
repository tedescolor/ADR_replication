# Notes

## Data
1000 rows, x0 ~ U(0,1). For x0 < ~0.5, y follows one smooth curve (min ~ -0.8 at x=0.2) with
Gaussian noise whose sd falls from ~0.4 to ~0.2. Around x0 ~ 0.55 the data split into two branches
of roughly equal weight: an upper branch rising almost linearly to ~2.5 at x=1, and a lower
branch that continues the original curve (bump to ~0.1 at 0.7, down to ~-1 at 1), with sd ~0.1.
Within-branch residuals look Gaussian (excess kurtosis ~0).

## Validation
Direct MLE (BFGS, several random restarts). Validation is 5-fold CV repeated twice (10 folds of 200),
with every model refitted on each training split.

| model | params | train NLL | CV NLL |
|---|---|---|---|
| free 2-comp mixture, cubic means, log-linear sd, logistic weight | 12 | 0.189 | 0.205 |
| free mixture, quintic means, quadratic sd/weight | 18 | 0.182 | 0.221 |
| shared quintic + softplus-hinge gap (lin+quad), log-linear sd | 13 | 0.185 | 0.202 |
| shared quartic + hinge gap (lin+quad), log-quadratic sd | 13 | 0.186 | 0.2009 |
| same, Student-t noise | 14 | 0.186 | 0.2012 |
| same, cubic gap | 14 | 0.185 | 0.2055 |
| shared cubic mean | 12 | 0.205 | 0.2158 |
| separate sd per branch / x-dependent weight | +2..3 | ~same | worse |
| **shared quartic + linear softplus-hinge gap, log-quadratic sd, constant weight** | **12** | **0.186** | **0.2001** |

## Final program
p(y|x) = w N(m1(x), sd(x)) + (1-w) N(m1(x) + a*softplus-hinge(x; s, h), sd(x)),
with m1 a quartic polynomial, log sd quadratic, and w constant (0.475). 12 fitted constants
(plus a few structural 1.0s), refitted on all of the data. I chose it because it had the best CV score
and the fewest parameters. The richer variants did not improve CV.
CV standard error is ~0.016 per fold. The full-data fit should do about as well as CV or slightly better.

expected_nll: 0.200
