# Notes

## Data
1000 rows, x0 in (0,1). Scatter plot: a single noisy curve for x0 < ~0.5 that splits
into two branches (one rising to ~2.5, one falling to ~-1 at x0=1). Noise shrinks
with x0 (sd ~0.38 at x0=0 to ~0.09 at x0=1). Standardised residuals look Gaussian
(a Student-t fit drove df to ~60).

## Validation
5-fold CV repeated with 2 shuffles (10 fits per candidate), MLE by BFGS (+Nelder-Mead
polish) warm-started from the full-data fit. Reported number = mean held-out NLL.

## Candidates (CV NLL)
- 2-normal mixture, sigmoid gate on x, separate poly means/log-sds (14-26 params): 0.249-0.289
- 2-normal mixture, Chebyshev means, separate log-sds, logit weight free: 0.248-0.255
- same with shared sd and weight fixed at 0.5: (4,4, logsd quad) 0.2395, 13 params
- symmetric form y = g(x) +/- h(x): g quartic, h = u(a+bu), u = max(x-c,0),
  shared log-quadratic sd, weights 0.5: **0.2390, 11 params** (train 0.2210)
- variants: h linear hinge 0.248; g cubic 0.243; g degree 5/6 0.240/0.241;
  log-sd linear 0.242 / cubic 0.240; Student-t noise 0.240.

## Choice
The symmetric hinge model: fewest parameters among the best CV group, and it
encodes the visible structure (one curve splitting into two at x ~ 0.52).
Constants are rounded to 6 significant figures (no change in train NLL).

expected_nll: 0.24
