# Notes

Data: 250 rows, y vs x0 in [0,1]. Scatter: a single noisy wavy curve for x<0.5 (noise sd ~0.3-0.45),
which splits after x~0.5 into two thin branches: a rising line (to ~2.5 at x=1) and a falling branch
(to ~-0.7 at x=1). So: a 2-component Gaussian mixture with x-dependent means, scales and weight.

Validation: 5-fold CV, repeated with 2-3 random fold assignments (held-out mean NLL of models refit
on each training split, multi-start L-BFGS/BFGS).

Tried (z = 2*x0-1; means poly deg dm, log-sd poly deg ds, K comps, logistic gate linear in z):
- (dm,ds,K) = (3,1,2): CV 0.234 (2 reps) / 0.226 (3 reps, other folds)
- (4,1,2): 0.242; (5,1,2): 0.272; (4,2,2): 0.318; (3,2,3): 0.387 -> extra flexibility overfits.
- Structured "sine + line" mixture (sine-shaped branch with log-linear sd, linear upper branch,
  logistic gate): CV 0.244; with x-dependent sd for line 0.256; Student-t 0.247; quadratic log-sd 0.241.

Final: (3,1,2) polynomial mixture fitted on all data (tiny 1e-3 ridge); 14 constants
(+2 for z = 2x-1). Train NLL 0.144, CV 0.226-0.234. Estimate for new data a bit above the CV mean.

expected_nll: 0.24
