# Notes
Data: y vs one x0. Mean is nonlinear in x0 and, for x0 > ~0.8, y is bimodal (a lower branch continuing
the curve and an upper branch near 2). Spread grows with x0.

Model: 2-component mixture of Normals. Lower component: quartic mean in x0, log-sd linear in x0.
Upper component: linear mean, log-sd linear in x0. Mixture weight is sigmoid(linear in x0).
13 constants. Fitted by ML (BFGS, multiple starts).

Validation: 5-fold CV (shuffled, seed 0), refitting each fold. Mean NLL: quartic 0.241,
cubic 0.261 (a quintic fit failed to converge, worse). Training NLL 0.177. Chose quartic.
The CV figure is slightly optimistic because the degree was chosen on the same folds.

expected_nll: 0.25
