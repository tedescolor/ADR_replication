# Notes
Data: 250 rows, 1 covariate. Mean of y is non-monotone in x0 (dips around 0.25, rises after 0.5); spread grows strongly toward x0=1 and is also larger near 0.
Validation: 3x repeated 10-fold CV of mean NLL, refitting by MLE each fold.
Candidates (CV NLL): Normal poly-mean deg 3-6 x log-sd poly deg 1-3; Student-t; SkewNormal (const / linear shape).
Constant log-sd degree 1: ~0.72; deg 2: ~0.596-0.60; higher degrees, t and skew-normal did not help (0.597-0.62).
Final: Normal(cubic mean, exp(quadratic log-sd)), 7 constants, fit by MLE on all data (train NLL 0.576).
CV NLL ~0.596-0.60 (note: model selection among ~30 candidates adds slight optimism).
expected_nll: 0.61
