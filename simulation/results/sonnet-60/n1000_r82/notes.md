# Notes
Data: 1 covariate x0 in [0,1]; y has a wiggly mean (quartic-like) and heteroscedastic spread for x<0.6,
and for x>~0.65 splits into two branches (lower continuing the curve, upper rising ~linearly).
Model: two-component Gaussian mixture. Component A: quartic mean in u=x-0.5, log-sd quadratic.
Component B: linear mean, constant sd. Weight of B = sigmoid(linear in u). 13+... ≈ 12 fitted params,
~25 literals total. Fitted by ML (L-BFGS, 6 starts) on all 1000 rows.

Validation: 5-fold CV (shuffled, seed 1), refitting on each fold. Mean held-out NLL:
- A mean cubic, sd linear, B linear: 0.291
- A quartic, sd linear: 0.278
- A quartic, sd quadratic (chosen): 0.275
Larger configs (more terms) were not completed within time, so not evaluated.
In-sample NLL of final model: 0.258 (verified through model.py).

expected_nll: 0.28
