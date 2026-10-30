# Notes

## Data
250 rows, x0 in [0,1]. For x < ~0.5, y lies on a single noisy curve (sd ~0.3). After that it
splits into two branches: an upper branch rising almost linearly to ~2.6, and a lower branch
that bends down to ~-1. The two are about equally populated (~47% upper), and the noise
shrinks with x (sd ~0.05–0.1 near x=1).

## Validation
5-fold CV, repeated twice with different shuffles (500 held-out predictions), refitting each
model by maximum likelihood on every training split (BFGS + Nelder-Mead, warm-started from
the full-data fit plus jittered restarts). Paired per-point differences were used to compare
models. One standard error of the mean CV NLL is about 0.06, but paired differences are
much tighter (~0.01).

## Candidates (CV mean NLL)
- 2-comp Normal mixture, free polynomial means (quartic/quadratic), separate log-linear/quad
  scales, sigmoid gate with free location/slope/max weight (16 params): 0.170. The gate
  jumped to x=0.32, which is overfit.
- Same with weight capped at 0.5: 0.163; other polynomial degrees: 0.171–0.182.
- Same with Student-t components (18 params): 0.156.
- "Pitchfork": lower branch = main mean − (b1·u + b2·u²), u = max(x−c, 0), gate 0.5·sigmoid(k(x−c)):
  - separate scales, quartic main curve (14): 0.183; with t (15): 0.180
  - shared scale exp(quadratic), quartic main curve (12): 0.157; with t (13): 0.148
  - **shared scale, cubic main curve, Normal (11 params): 0.149** (paired diff vs the quartic
    version −0.009 ± 0.008)

## Final program
The cubic-main-curve pitchfork with a shared scale and Normal components: 11 fitted constants
(13 literals counting the 0.5 and 0.0), split point c = 0.527. It ties with the best t
variant on CV, but has fewer parameters and a more plausible structure (both branches
leave the main curve continuously at c), so I picked it. Its in-sample NLL is 0.104.

## Estimate
CV gave 0.149. Choosing the best of ~10 candidates biases that down a little, while the final
model is fit on 25% more data than each CV fit, which helps a little. Net estimate ~0.15.

expected_nll: 0.15
