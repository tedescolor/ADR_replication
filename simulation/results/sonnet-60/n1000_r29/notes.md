# Notes
Data: 1 covariate x0 in [0,1]. Mean of y is non-monotone in x0 (dips near 0.25, rises after); sd grows with x0.
Candidates: Normal / StudentT, polynomial mean (deg 3-6) and polynomial log-sd (deg 1-3), fit by ML (BFGS).
Validation: 10-fold CV mean NLL. Linear log-sd ~0.744; quadratic log-sd ~0.631 for all mean degrees 3-6;
StudentT gave no gain (tails are not heavy). Chose simplest: Normal, cubic mean, quadratic log-sd (7 constants + offsets).
Final fit on all data (train NLL 0.625); CV NLL 0.632.

expected_nll: 0.633
