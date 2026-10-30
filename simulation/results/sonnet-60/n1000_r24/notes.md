# Notes
Data: 1 covariate x0 in [0,1]. Mean of y is nonlinear in x0; spread grows at high x0, where y becomes
bimodal (a lower branch near -0.5 and an upper branch near 2) with the upper branch's weight rising in x0.

Model: 2-component Normal mixture. Component 1 has a quartic mean in x0 and a quadratic log-sd.
Component 2 has a quadratic mean and a linear log-sd. The upper-branch weight is sigmoid(quadratic in x0).
16 fitted constants, fitted by maximum likelihood (BFGS, 6 random starts).

Validation: 5-fold CV (fixed shuffle) over polynomial degree configurations (refit of all parameters in each fold).
CV NLL: 0.2755 (12 params), 0.253 (14), 0.2249 (chosen, 16 params), 0.229-0.230 for larger ones (17-18 params).
In-sample NLL of the chosen fit: 0.212. Selection among ~15 configs adds slight optimism.

expected_nll: 0.23
