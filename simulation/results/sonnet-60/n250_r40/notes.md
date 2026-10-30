# Notes
Data: 250 rows, 1 covariate. Mean of y is nonlinear in x0 (high near 0, dip near 0.2, rises after 0.5);
spread grows strongly with x0 and is slightly larger near 0 (U-shaped log-scale).

Validation: 10-fold CV (3 repeats for the final choice), refitting by ML inside each fold.
Candidates (CV NLL): Normal with poly-mean degree 3..6 and log-sd degree 1..3; StudentT and SkewNormal variants;
sine+linear mean. Linear log-sd ~0.795; quadratic log-sd ~0.648 (big gain); cubic log-sd/higher mean degrees no gain;
StudentT (4,2) 0.6485 (df large, no gain); SkewNormal 0.654-0.657 (worse); sine mean 0.652.

Final: Normal(quartic poly mean in x0, sd = exp(quadratic in x0)); 13 constants.
Training NLL 0.6215; CV NLL 0.648.

expected_nll: 0.65
