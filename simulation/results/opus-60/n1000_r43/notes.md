# Notes

## Data
1000 rows, x0 in (0,1). Scatter (work/scatter.png) shows a single noisy curve for x < ~0.4
(minimum ~ -0.85 at x=0.2) that splits into two branches: an upper one rising to ~2.5 at x=1,
and a lower one staying near 0 then falling to ~-1 at x=1. Noise shrinks with x
(sd ~0.4 at x=0, ~0.08 at x=1). Per-branch standardized residuals look Gaussian
(skew/kurtosis near 0).

## Candidates (10-fold CV mean NLL, MLE via BFGS per fold)
- 2-component Gaussian mixture, independent polynomial means (deg 3-7), log-sd polynomials,
  free/constant weight: best ~0.212-0.214 (deg 5 means). Free weight, separate sds: no gain.
- Shared linear sd, weight 0.5: ~0.212.
- Structured: upper = polynomial U(x); lower = U(x) - a*max(x-k,0)^p; shared linear sd;
  weight 0.5. Big gain: ~0.205-0.207. p fixed at 2 is as good as free p (fitted 1.8).
  Free weight, exp-linear sd, extra lower-branch terms: no gain.
  Degree of U: 3 -> 0.2071, 4 -> 0.2076, 5 -> 0.2080, 6 -> 0.2089 (mean over 3 repeated
  10-fold splits). Chose cubic U (8 constants fitted).

## Final program
Mixture(0.5/0.5) of Normal(U(x), sd) and Normal(U(x) - 8.39*max(x-0.353,0)^2, sd),
U cubic, sd = 0.400 - 0.319 x (floored at 0.02). 8 fitted constants (+ weights/floor,
13 literals total). Training NLL 0.1975; CV NLL 0.205-0.208 (fold SE ~0.019).

expected_nll: 0.207
