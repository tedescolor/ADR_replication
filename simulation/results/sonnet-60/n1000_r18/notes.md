# Notes
Data: y vs x0 in [0,1]. Single noisy branch for x<0.5, splitting into two branches (rising, falling) with growing spread for x>0.55.
Model: 2-component Gaussian mixture; each component mean = quartic poly in x0, log-sd linear; weight = sigmoid(cubic). 27 constants (+ a few).
Validation: 5-fold CV (refit incl. 6 random restarts per fold). Compared (mean deg, sd deg, weight deg): (3,1,2) 0.226, (4,1,3) 0.222, (4,2,3) 0.246, (5,2,4) 0.256 (higher-degree ones overfit/unstable). Chose (4,1,3). A further batch of variants was not completed in the time budget.
Full-data train NLL 0.206; CV NLL 0.222.
expected_nll: 0.23
