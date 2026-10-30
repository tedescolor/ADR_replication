# Notes

Data: 1000 rows, x0 ~ U(0,1). Scatter shows one noisy curve (sd ~0.25-0.4) for small x0
that splits into two branches: an upper near-linear branch rising to ~2.5 at x0=1 and a
lower branch curving down to ~-1 at x0=1, roughly equal mass, branch noise ~0.1.

Validation: 5-fold CV (polynomial family: repeated twice), mean held-out NLL.

1. 2-component Gaussian mixture, means / log-sds / mixing logit polynomial in t=2*x0-1:
   - degrees (3,2,2): train 0.152, CV 0.172 (17 params)
   - (5,2,2): train 0.145, CV 0.173;  (7,3,3): train 0.140, CV 0.182 (overfits)
   - (3,1,0): train 0.165, CV 0.180
2. Structured "shared cubic base + hinge branches" mixture (work/hinge.py): both
   components share a cubic mean base(t); branch k adds u_k*h + v_k*h^2 with
   h = max(x0 - c, 0) (c fitted, = 0.383); quadratic log-sds; constant weight.
   16 params; train 0.1499, 5-fold CV 0.165 (folds 0.19, 0.13, 0.20, 0.22, 0.08).
   -> chosen: better CV than the free polynomial mixture with fewer parameters, and it
   encodes the visible structure (one curve that splits).

Final model.py: 2-component Normal mixture as in (2), refit on all data. 16 fitted
constants + 2 structural (2.0, 1.0 in t) + 0.0 in the hinge. Fold-to-fold spread is large
(0.08-0.22), so I add a small margin to the CV estimate.

expected_nll: 0.17
