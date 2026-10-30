# Notes

## Data
250 rows, x0 in (0,1). Scatter shows a single noisy oscillating curve for x < ~0.55
(min ~-0.8 near x=0.22), then a split into two branches: a tight, nearly linear upper
branch rising to ~2.45 at x=1 (slope ~5), and a lower branch continuing the curve down to
~-1 at x=1. Spread shrinks with x (sd ~0.44 at x=0 to ~0.12 at x=1).

## Models
Two-component Gaussian mixture, weight sigmoid(a+b x):
- comp 1: polynomial mean (deg 3/4/5) of the main curve, log-sd polynomial;
- comp 2: linear mean (upper branch).

Validation: 10-fold CV repeated 2-3 times (different shuffles), refit by ML (BFGS,
several starts) in each fold; reported mean held-out NLL per point (SE ~0.054, but the
differences between models are paired, so they are more precise than that).

| variant | params | train NLL | CV NLL |
|---|---|---|---|
| cubic mean, log-linear sd1, const sd2 | 11 | 0.261 | 0.309-0.313 |
| quartic, same | 12 | 0.253 | 0.305-0.310 |
| quintic, same | 13 | 0.253 | 0.318 |
| quartic, quadratic log-sd1 | 13 | 0.251 | 0.319 |
| quartic, log-linear sd2 too | 13 | 0.242 | 0.313 |
| sinusoid mean instead of quartic | 12 | 0.256 | 0.311 |
| quartic, Student-t comp 1 | 13 | 0.250 | 0.303 |
| quartic, sd2 = sd1 * exp(d) | 12 | 0.244 | 0.297 (d = -0.08, about 0) |
| **quartic, one log-linear sd shared by both comps** | **11** | **0.244** | **0.293** |

Sharing one scale across the two components was the best of these first variants. I then
tried other weight functions (10-fold CV x2 shuffles; on this scheme the variant above
scores 0.297):

| variant | CV NLL |
|---|---|
| weight = cap*sigmoid(k(x-c)), all free (k went to ~7000: a step at 0.32, cap 0.507) | 0.274 |
| sd linear (not log) in x | 0.310 |
| Student-t components, shared sd | 0.298 |
| quartic mean, weight = 0.5 * step(x > c), c profiled on a grid | 0.269 |
| cubic mean, weight = 0.5 * step(x > c) | 0.265 |
| quartic mean, weight = 0.5 * sigmoid(50 (x - c)) | 0.267 |
| **cubic mean, weight = 0.5 * sigmoid(50 (x - c))** | **0.264** |

So the data look like a bifurcation: a single noisy curve that splits at x ~ 0.34 into
two equally likely branches with the same noise level. The threshold c was refit inside
every CV fold, so these CV numbers are honest.

## Final program
w = 0.5 * sigmoid(50 (x - 0.34));
comp 1 = Normal(cubic(x), s), comp 2 = Normal(5.078 x - 2.45, s), s = exp(-1.413 x - 0.815).
I chose this one because it had the lowest CV score and the fewest parameters (8 fitted
coefficients + c; k=50 and the 0.5 cap are fixed). 12 numeric literals. The steep sigmoid
is preferred over a hard step because it is more robust to new points near the switch.
Training NLL is 0.221; CV NLL is 0.264 (SE of the mean ~0.05). The CV number already
includes the threshold search, but the final choice was made among about 15 variants,
so I add a small margin.

expected_nll: 0.27
