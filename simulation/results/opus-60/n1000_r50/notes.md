# Notes

## Data
1000 rows, x0 in (0,1). A scatter plot shows one noisy curve for x < ~0.5 (it dips to about -0.7 at x≈0.25, sd ≈0.3–0.4)
that splits into two branches: an upper branch rising almost linearly to about 2.5 at x=1, and a lower
branch that curves down to about -1. Noise shrinks as x grows (branch sd ≈0.1–0.15).

## Candidates (5-fold CV, same folds, mean held-out NLL; SE ≈0.023)
- Free 2-component Gaussian mixture, polynomial means/log-sd/logit-weight in t=2x-1:
  (deg 3,1,1) 0.244; (4,2,1) 0.236; (5,2,2) 0.236.
- Structured "branching" mixture: shared polynomial base b(t); component k mean = b + h_k1*r + h_k2*r^2 with
  r = max(x-c, 0) (fitted split point c ≈0.35); quadratic log-sd per component; logistic weight in t:
  - base cubic, linear branches: 0.257 (branches need curvature)
  - base cubic, const weight, linear log-sd: 0.234
  - base cubic, quadratic log-sd, logistic weight: **0.232** (17 params)
  - base quartic, same: 0.2315 (18 params); base quintic: 0.234
- Below the split both components have the same mean and different sd, so they act as a scale mixture.

## Choice
I chose the cubic-base branching mixture with 17 constants (+ the fixed 0.0/1.0/2.0 helpers), fitted by ML on all data
(train NLL 0.210). It matches the best CV score with fewer parameters, and the quartic base gives no real gain.
The train–CV gap is about 0.02, so I estimate the NLL on new data at about 0.232.

expected_nll: 0.232
