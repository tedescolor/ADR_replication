# Notes
Data: 1 covariate x0 in [0,1]. Lower curve is cubic-like in x with growing noise; for x>~0.6 a second
upper branch (roughly linear, rising to ~2.5) appears and takes up more weight as x grows.
Model: 2-component Normal mixture; comp A mean cubic in x, sd=exp(linear in x); comp B mean linear, constant sd;
weight of B = sigmoid(linear in x). 11 fitted parameters, fit by ML (L-BFGS, multi-start).
Validation: 5-fold CV (shuffled) comparing A-degree 3/4/5 and B-degree 1/2: CV NLL 0.2747 (3,1), 0.2748 (4,1),
0.2707 (5,1), 0.2825 (4,2). Differences are within noise, so chose the simplest (3,1). Train NLL 0.214.
CV estimate is slightly pessimistic (fit on 200 rows).
expected_nll: 0.28
