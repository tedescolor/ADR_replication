# Notes
Data: y vs x0 in [0,1]. Mean is nonlinear in x0 and spread varies. For x0 > ~0.65 a second branch
(y ~ 1.2..2.6, rising roughly linearly in x0) appears next to the lower branch, so the conditional density is bimodal there.

Model: 2-component Normal mixture.
- Component A: mean = quartic polynomial in u = x0-0.5, sd = exp(linear in u).
- Component B: mean = linear in u, constant sd.
- Weight of B = sigmoid(linear in u).
12 fitted parameters, about 17 literals in total, fitted by maximum likelihood (L-BFGS, 3 random starts).

Validation: 5-fold CV (fixed random folds) over 7 candidate configurations (polynomial degrees for the means,
log-sd and weight logit). All scored 0.256-0.265 mean NLL, with a fold SE of about 0.035, so they are indistinguishable.
I chose the simplest one (CV NLL 0.259). The final fit on all data has in-sample NLL 0.2495.

expected_nll: 0.26
