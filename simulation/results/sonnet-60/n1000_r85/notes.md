# Notes
Data: 1 covariate x0 in [0,1]. The mean of y is nonlinear: it dips near x=0.2, then rises and plateaus. The spread grows strongly with x.
Candidates: Normal or StudentT, with a polynomial mean (degree 3-6) and log-sd polynomial (degree 1-3).
Validation: 10-fold CV, one random split, mean NLL on held-out folds.
- log-sd linear: about 0.766
- log-sd quadratic: about 0.628-0.630
- Student-t gave no gain; the mean degree barely mattered.
Final: Normal, quartic mean, log-sd quadratic, 8 constants, fit by maximum likelihood on all the data.
Training NLL 0.621; CV NLL about 0.628.

expected_nll: 0.63
