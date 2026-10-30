# Notes
Data: 1 covariate x0 in [0,1], y with a wiggly nonlinear mean and a U-shaped spread (sd large near x=0, small ~0.3 mid, large near x=1).
Validation: 5-fold CV (fixed shuffle seed) on 1000 rows. Candidates: Normal vs StudentT, polynomial mean degree 3-6, polynomial log-sd degree 1-3.
CV NLL: log-sd linear ~0.77-0.78 (bad); log-sd quadratic: mean deg3 0.635, deg4 0.633, deg5 0.633, deg6 0.630; cubic log-sd no better.
StudentT gave no gain over Normal (residuals ~Gaussian), so Normal was kept.
Final: Normal(mean = degree-6 polynomial in x0, sd = exp(quadratic in x0)); 10 fitted constants plus exponents, fit by ML on all data.
Train NLL 0.623, CV NLL 0.630. The degree-6 choice was picked from a small grid on the same CV folds, so there is slight selection optimism.
expected_nll: 0.635
