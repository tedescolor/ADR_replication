# Notes
Data: y vs x0 shows two branches splitting near x0≈0.45 (upper rises to 2.5, lower falls), heteroscedastic.
Model: 2-component Gaussian mixture. Means are quartic polynomials in x0; log-sd are quadratics;
the upper-component weight is sigmoid(quadratic). About 20 constants. Fit by ML (L-BFGS, multiple starts).
Validation: 5-fold CV over (mean degree, log-sd degree, weight degree) configs.
CV NLL: (4,2,2) .2148, (5,2,3) .2130, (6,2,3) .2122, (6,3,3) .2164, (7,3,4) .2212, (5,3,3) .2121.
I chose (5 coefficients for the mean, 3 for log-sd, 3 for the weight logit), refit on all the data.
Train NLL 0.195; CV NLL 0.212.
expected_nll: 0.215
