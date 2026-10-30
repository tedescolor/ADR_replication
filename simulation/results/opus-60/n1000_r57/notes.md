# Notes

## Data
1000 rows, x0 in [0,1], y. Scatter: one noisy curve for x < ~0.52 (dip to -0.8 near x=0.25,
sd ~0.4 falling to ~0.25), which then splits into two branches of about equal weight:
an upper branch rising almost linearly to ~2.5 at x=1, and a lower branch with a hump (~0.3 at x=0.7)
falling to ~-0.9 at x=1. Branch noise sd is ~0.1.

## Validation
5-fold CV repeated twice (10 folds in total, same folds for every model), with a full refit
(multi-start L-BFGS MLE) in each fold. Fold-level SE is about 0.024, but because every model
uses the same folds, the comparisons between models are paired and much tighter than that.

## Candidates (CV mean NLL)
- Two-Gaussian mixture, free polynomial means of degree d, quadratic log-sd, logistic-linear weight:
  d=3: 0.2229, d=4: 0.2235, d=5: 0.2260.
- Structured split model (chosen family): shared base curve b(x) (polynomial in t=2x-1),
  plus a softplus ramp r(x)=h*softplus((x-c)/h) times a per-branch polynomial a_k(t):
  m_k = b + r*a_k; log-sd_k polynomial; constant weight. Config (deg b, deg a, deg log-sd, deg weight):
  (3,1,2,0) 0.256 (optimizer trouble in some folds), **(4,1,2,0) 0.2196**, (5,1,2,0) 0.2212,
  (4,1,2,1) 0.2212, (4,1,1,0) 0.2244, (4,1,3,0) 0.2241, (4,2,3,0) 0.2248, (4,0,2,0) 0.2225.
- PIT of the chosen model: skew -0.12, excess kurtosis 0.00, and calibrated within each x-decile,
  so I kept Gaussian components (no Student-t or skewed noise).

## Final
(4,1,2,0) split model fitted on all data: 18 fitted parameters (24 literals in the file), train NLL 0.2002.
The fitted changepoint is c=0.509 with ramp width h=0.013, and the weight is 0.53/0.47.
I chose it because it had the best CV and is structurally faithful to the data: one curve that bifurcates.
Expected held-out NLL is about the CV value; the model fitted on all data should do slightly better
than the fold models, so I report 0.22.

expected_nll: 0.22
