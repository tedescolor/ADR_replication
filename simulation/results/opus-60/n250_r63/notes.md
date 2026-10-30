# Notes

## Data
250 rows, x0 in (0,1). For x0 < ~0.6, y is unimodal around a wavy curve (≈ -0.4 at 0, min ≈ -0.8 near 0.2,
≈ +0.4 near 0.6). Above ~0.65 y splits into two branches: a lower one that keeps following the wavy
curve down to ≈ -0.85 at x0=1, and a tight upper branch that rises roughly linearly to ≈ 2.4.
The noise around the main curve shrinks as x0 grows.

## Models tried (10-fold CV, 2 repeats, MLE refit in every fold, mean NLL per point)
2-component Normal mixture: polynomial mean for each component, polynomial log-sd, logistic weight.
Degree tuple (deg+1 for m1, m2, logs1, logs2, logit w):
- (4,2,1,1,2) 0.352; (5,2,1,1,2) 0.353; (6,2,1,1,2) 0.325; (5,3,1,1,2) 0.339
- heteroscedastic main component: (5,2,2,1,2) **0.241**; (5,3,2,1,2) 0.262; (5,2,3,1,2) 0.249;
  (5,2,2,2,2) 0.244; (4,2,2,1,2) 0.284; (4,2,3,1,2) 0.289; constant upper branch (5,1,2,1,2) 0.422
- Sine mean for the main branch, m1 = a + b sin(c x + d) + e x, with the same linear upper branch,
  linear log-sd for the main branch, constant sd for the upper branch and logistic weight in x:
  12 parameters, train 0.170, **CV 0.225**. This is the best result, and it uses the same number of
  constants as the best polynomial model.

## Final program
Mixture of Normal(a + b sin(c x + d) + e x, exp(q0 + q1 x)) and Normal(p0 + p1 x, exp(r0)),
weight sigmoid(w0 + w1 x). 12 constants, fit by MLE on all data. Passes the contract check.

## Expected NLL
The CV score is 0.225. Because I chose the model family partly by looking at CV over about a dozen
candidates, I add a small allowance for selection optimism.

expected_nll: 0.24
