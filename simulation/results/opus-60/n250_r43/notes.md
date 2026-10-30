# Notes

Data: 250 rows, x0 in [0,1], y in [-1.2, 2.7]. Scatter shows a wavy lower branch
(min ~-0.8 at x~0.25, max ~0.2 at x~0.65, falls to -1 at x=1) and an upper branch that
splits off around x~0.45 and rises almost linearly to ~2.4 at x=1. At low x the spread is
wider (sd ~0.3), with apparent sub-clusters, i.e. some bimodality there too.

Validation: 10-fold CV (two different random fold assignments), refitting all
parameters by maximum likelihood in each fold (warm-started from the best full-data fits).

Candidates (CV mean NLL, two seeds):
- hand-built: lower = linear + sinusoid, upper = hinge quadratic, sigmoid gate: 0.197 / 0.179
- same + sinusoidal wiggle on upper branch: 0.171 / 0.160
- upper = lower + hinge quadratic (weights ~0.5 everywhere): 0.165 / 0.171
- generic 2-Gaussian mixture, Chebyshev degree-5 means, linear log-sd, linear logit
  weight (18 params): **0.157 / 0.159**
- same with degree-7 means, quadratic log-sd and logit (25 params): 0.214 (overfits)
- 3-Gaussian mixture, degree-5 means (28 params): train NLL -0.03, warm-started CV 0.142,
  BUT the third component had sd ~0.001 and threaded a degree-5 curve exactly through 10
  scattered points (degenerate spike). The warm-started CV leaks: fold fits start from a
  full-data solution that already passes through the held-out points. Rejected.
- 3-Gaussian with sd floor 0.05 (sd = 0.05 + exp(lin)): CV 0.242. Rejected.
- 2-Gaussian variants: degree-4 means 0.182; degree-6 0.163/0.197 (unstable); constant sd
  0.214; quadratic logit 0.177; constant weight 0.179/0.189.

Honest CV for the chosen model (6 random inits per fold, no warm start from full-data fit):
0.201 -- pessimistic, since with only 6 inits the optimiser often lands in a worse local
optimum (full-data runs found the 0.092 optimum only in a minority of starts). Warm-started
CV (0.157-0.159) is optimistic. The truth is likely in between.

Final model: model.py = 2-component Gaussian mixture; component means are degree-5
polynomials in z = 2*x0 - 1, log-sds linear in z, mixing logit linear in z.
18 fitted constants (+2 for z, +1 for the reference logit 0.0 = 21 literals).
Full-data train NLL 0.092.

expected_nll: 0.18
