# Notes

## Data
250 rows, x0 in (0,1). Scatter: a single noisy wavy branch for x < ~0.55 (sd ~0.3 at x=0,
shrinking to ~0.1 at x=1). At x ~0.6 it bifurcates: an upper, concave, nearly linear branch
rising to y ~2.5 at x=1 (sd ~0.1), and the continuation of the wavy branch curving down to
y ~ -1. Above x ~0.65 the points split roughly 50/50 between the two branches.

## Validation
5-fold CV, repeated twice with different shuffles (10 fits per model), out-of-fold mean NLL.
Fits were MLE (Nelder-Mead then BFGS, several restarts). Caveat: for the sine models the
fold fits are started from the full-data optimum, which leaks a little (initialisation only).

## Candidates (CV NLL / in-sample NLL)
- 2-Normal mixture, poly lower branch (deg 3-6), linear upper, log-linear scales, logistic
  gate: best quintic 0.260 / 0.191; Student-t components 0.268; higher degree overfit.
- Lower mean a+bx+c sin(fx+ph), linear upper, logistic gate: 0.257; with quadratic upper
  branch: 0.251.
- Plateau gate w = wmax*sigmoid(k(x-x0)): with constant sd_B 0.260; with log-linear sd_B
  0.2435; with log-quadratic sd_A 0.273 (overfits).
- **Final:** same as the previous model but with the plateau fixed at 0.5 (matches the visible
  50/50 split, one fewer parameter): **0.2383** CV, in-sample 0.163.

## Final program
Mixture of two Normals:
- A (lower branch): mean 0.3057 - 1.2225 t + 0.7922 sin(5.8137 t + 3.6347), sd exp(-0.899 - 1.298 t)
- B (upper branch): mean quadratic in t, sd exp(0.7305 - 3.223 t)
- weight of B: 0.5 * sigmoid(16.98 (t - 0.586))
That is 15 literals: 14 fitted constants plus the fixed 0.5 plateau.

## Expected NLL
The CV estimate is 0.238. I raised it a bit to cover the small initialisation leak, the
selection over about 15 candidates, and the CV noise with n=250 (fold-to-fold SE ~0.03).

expected_nll: 0.25
