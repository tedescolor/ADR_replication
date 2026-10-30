# Notes
Data: 1 covariate, nonlinear mean (dip near x=0.2, rise to x~0.9), sd grows strongly with x.
Model: Normal(mu(x), sd(x)); mu cubic polynomial in (x-0.5); log sd quadratic. 7 fitted constants (+2 literals), fit by penalised ML.
Validation: 5-fold CV x3 repeats over grid of mean degree 3-6, log-sd degree 1-3, Normal vs StudentT, plus skew-normal.
CV NLL: Normal(3,2)=0.658; Student-t variants 0.652-0.66 (noise-level, not chosen); skew-normal 0.664-0.683; sd-linear ~0.79.
Chose simplest Normal(3,2). Train NLL 0.628; CV 0.658.
expected_nll: 0.66
