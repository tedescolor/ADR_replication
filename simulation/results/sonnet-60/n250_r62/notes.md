# Notes
Data: 250 rows, one covariate x0 in [0,1]. Mean of y is nonlinear (dips then rises), spread grows strongly with x0.
Candidates (maximum-likelihood fits, polynomial mean degree dm, polynomial log-sd degree ds, noise Normal / Student-t / skew-normal),
compared by 10-fold CV NLL (one random split, same folds for all):
- Normal dm=3, ds=2: 0.701 (best); t gives identical value (tails not heavy)
- dm=4 ds=2: 0.704; dm=3 ds=3: 0.704; dm=5 ds=2: 0.707; dm=2: 0.75; ds=1: 0.84; skew-normal no better (0.705 at dm=2, worse otherwise).
Final: Normal(cubic mean, exp(quadratic)) fit on all data; 7 constants (plus 0-exponent literals), train NLL 0.678.
CV used a single split; selection among ~20 candidates adds slight optimism, so I add a small margin.
expected_nll: 0.71
