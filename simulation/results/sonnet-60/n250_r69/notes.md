# Notes
Data: 250 rows, x0 in [0,1]. Mean of y is nonlinear (dip near x=0.25, rise after 0.5); sd is U-shaped (small mid-range, large at x>0.8).
Validation: 10-fold CV (seed 0) of MLE fits, mean held-out NLL. Candidates: Normal / StudentT / SkewNormal, polynomial mean degree 3-6, log-sd polynomial degree 1-4.
Key results: log-sd linear 0.714, quadratic 0.586 (big gain); degree 4 mean best; StudentT and SkewNormal gave no gain (0.586 / 0.595); log-sd degree 4 0.5835 (within noise, not chosen).
Final: Normal(quartic mean in x0, exp(quadratic) sd): 8 constants + a few structure literals. Train NLL 0.553; CV NLL 0.586 (selection among ~24 configs adds slight optimism).
expected_nll: 0.60
