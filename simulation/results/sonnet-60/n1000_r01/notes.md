# Notes
Data: 1 covariate x0 in [0,1]. Scatter shows two branches: a wavy lower branch over all x, and an upper, roughly linear branch (y≈-2.3+4.9x) that emerges around x≈0.4. Noise shrinks with x.
Model: 2-component Normal mixture. Lower branch has a quartic mean and log-sd linear in x. Upper branch has a linear mean and log-sd linear in x. Weight of the upper branch is sigmoid(linear in x). 15 constants.
Validation: 5-fold CV (shuffled), refitting by ML each fold. Lower-branch degree 3..7 and upper-branch degree 1..2 compared. Degree 4 with a linear upper branch gave CV NLL 0.262, and degree 3 was worse (0.275-0.281). Higher degrees gave no gain. Final fit on all data has train NLL 0.248.
expected_nll: 0.265
