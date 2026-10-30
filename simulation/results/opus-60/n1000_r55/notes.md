# Notes

Data: 1000 rows, x0 in (0,1). For x < ~0.45 y forms one noisy Gaussian band (sd ~0.4 at x=0
falling to ~0.25); above that it splits into two thin branches (sd ~0.1): an upper branch
rising to 2.5 and a lower branch peaking near 0.1 at x~0.7 and falling to -0.9 at x=1.
About 50% of points fall on each branch at every x > 0.6.

Validation: 10-fold CV of the mean held-out NLL (scored over all rows), with parameters refit by
BFGS MLE in each fold (warm-started from the full-data fit). Differences under ~0.003 are
within noise.

Tried (CV NLL):
- 2-Gaussian mixture, polynomial means of degree 3/4/5, log-sd linear/quadratic, logistic
  weights: 0.194-0.198 (degree 3 best; higher degrees overfit).
- Shared sd across components: 0.191-0.196.
- "Pitchfork" C: w = 0.5, upper mean = cubic a(x), lower mean = a(x) + max(x-c,0)^2 (p + q x),
  shared sd = exp(u + v x). 9 constants, CV 0.1914, train 0.1815.
- Variants (separate sds, cubic split, quartic base, linear sd): 0.192-0.194.
- 3-repeat 10-fold CV of the finalists: C 0.1917; C + logistic weight (11 consts) 0.1909;
  signed split z|z| (branches cross) 0.1963; free cubic mixture with shared sd and logistic
  weight (12 consts) 0.1911.

Final choice: C (9 constants). The extra-weight variants gain <0.001, which is within CV noise,
so I kept the simpler, more structured program. Train NLL is 0.1815 and CV is 0.1914-0.1917; the
CV warm start biases it slightly low, so I add a small margin for the estimate.

expected_nll: 0.193
