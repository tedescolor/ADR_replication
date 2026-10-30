# Notes
Data: 1 covariate, y has nonlinear mean (dips then rises) and variance growing sharply at large x.
Validation: 5-fold CV, repeated 4-6 times with different shuffles, mean held-out NLL.
Tried Normal / StudentT / SkewNormal with polynomial mean (deg 2-5) and polynomial log-sd (deg 1-4).
CV NLL: sd linear ~0.71; sd quadratic ~0.56 (cubic mean); deg 2 mean 0.63; deg 4-5 mean slightly worse (0.57-0.58);
StudentT no gain (0.560); SkewNormal unstable for deg>=4, no gain. Higher-degree log-sd no gain.
Final: Normal(cubic mean, exp(quadratic)) fit by ML on all data, 7 constants. Train NLL 0.534, CV NLL ~0.563.
expected_nll: 0.57
