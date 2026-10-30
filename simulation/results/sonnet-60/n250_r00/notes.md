# Notes
Data: x0 in [0,1]; y bimodal for x0>0.65. Lower branch follows a sinusoid in x0 (period ~0.8);
upper branch is linear in x0 and appears with weight rising in x0.
Model: 2-component mixture. Component A: Normal(c + a*sin(w x + p), exp(l + s x)); component B: Normal(linear, 0.135);
weight of B = sigmoid(k0 + k1 x). 15 constants, fitted by ML (L-BFGS, multiple starts).
Validation: 5-fold CV (refit each fold). Constant-sd version 0.291; heteroscedastic A (final) 0.269. Train NLL 0.200.
CV NLL is the honest estimate (slightly pessimistic since the final fit uses all data).
expected_nll: 0.27
