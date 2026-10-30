# Notes

## Data
250 rows, x0 in (0,1). Scatter shows a smooth curved main branch (≈0 at x=0, min ≈-0.8 at x≈0.2,
rising to ≈2.5 at x=1) and, for x > ~0.6, a second falling linear branch (≈0.3 -> -0.8);
for high x the conditional is bimodal (kurtosis ≈ -2). Main-branch noise shrinks toward x=1.

## Models tried (2-component Normal mixture, MLE, multistart L-BFGS)
Validation: 10-fold CV repeated twice (different shuffles), refitting on each training fold.
Notation (deg+1 of main mean, log-sd main, lower mean, log-sd lower, gate):
- logistic gate, (4,2,2,1,2): train 0.365, CV 0.416
- (4,3,2,1,2): CV 0.423; (5,3,2,1,2): CV 0.426; (5,2,2,1,2): CV 0.424
- (4,3,2,2,2): train 0.322, CV 0.412
- plateau gate w = c*sigmoid(k(x-x0)), (4,3,2,2): train 0.276, CV 0.377  <- best
The fitted plateau gate is essentially a step at x≈0.596 with weight ≈0.5, i.e. the
lower branch appears abruptly. This beat every logistic-gate variant by ~0.035 nats in CV.

## Final program
Main: Normal(cubic mean, exp(quadratic) sd); lower: Normal(linear mean, exp(linear) sd);
weight 0.4973*sigmoid(3654*(x-0.596)). 17 constants. Fitted on all data.
Train NLL 0.276, CV NLL 0.377. Expected test NLL a little above CV (model selection
among ~7 variants, time-limited search).

expected_nll: 0.39
