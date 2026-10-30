# Notes

## Data findings
- x1..x24 are highly collinear and on the same scale as y (around 35). x0 is a different kind of variable and is only weakly related to y.
- Key finding: **y almost never exceeds x24**. The 99th percentile of y − x24 is −0.003 and its maximum is +0.019. x24 also behaves like the max of x1..x23 plus noise of about ±0.02. So x24 looks like a maximum taken over a set that includes y.
- The gap g = x24 − y is bimodal. About 22% of rows sit in a narrow cluster at g ≈ 0–0.1, where y is effectively the max. The rest form a broad bulk at g ≈ 0.2–2.
- Which mode a row falls in is predicted by m = x24 − max(x1..x23). When x24 is clearly above all the other covariates, the max must have come from y, so g is small. When m ≤ 0, g usually comes from the broad bulk.

## Model
A two-component mixture on y:
- the "near" component is SkewNormal(x24 − 0.008, 0.040, shape −3.0);
- the "below" component is SkewNormal(x24 − 0.085, scale, shape −22). This is essentially a half-normal below x24. Its scale is 0.93·sd^0.40, where sd = std(x1..x23);
- the weight is sigmoid(−2.86 + 8.26·sqrt(max(m, 0))).

It has 9 fitted constants plus 2 guard constants (0.0 and 0.01).

## Validation
- 10-fold CV with one fixed random permutation; every model was refit by MLE (L-BFGS) inside each fold.
- Baselines: marginal normal 1.059; linear regression with normal errors 0.634; linear regression with Student-t errors 0.601.
- Mixture with constant location and weight linear in m: 0.199. Putting all 23 covariates into the location made it worse (0.257), which is overfitting.
- Weight on sqrt(max(m, 0)): 0.156. Adding log(sd) to the "below" scale: **0.141** (SE 0.044). This is the final model.
- Further additions changed CV by at most about 0.004, within noise, so I did not use them. They were: location on mean(x1..23) − x24 or on (x15 + x16)/2 − x24 (best 0.138), a sd term in the weight, and a sd term in the near-component scale.
- Training NLL of the final model is 0.126.

expected_nll: 0.145
