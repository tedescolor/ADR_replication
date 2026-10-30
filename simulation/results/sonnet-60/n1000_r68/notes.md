# Notes
Data: y|x0 is bimodal for x0>0.65 (two branches); lower branch has a smooth non-monotone mean, upper branch rises to ~2.5.
Model: 2-component Gaussian mixture; component means polynomial in x (cubic, quadratic), log-sd linear in x,
mixing logit quadratic in x. 14 fitted parameters (ML, L-BFGS, multi-start).
Validation: 5-fold CV, compared configs (mean degree, sd form): (2,1,const sd) 0.405, (3,1,const) 0.317, (3,1,lin) 0.263,
(4,1,lin) 0.263, (3,2,lin) 0.186 (chosen), (4,2,lin) 0.189, (5,2,lin) 0.190. Train NLL of final fit 0.171.
Selection among ~12 configs gives slight optimism; I add a small margin.
expected_nll: 0.20
