# Notes
Data: 1 covariate x0 in [0,1]; scatter shows two branches (one rising to ~2.5, one falling to ~-1 at x=1) merging into a noisy blob at low x, with noise shrinking as x grows.
Model: 2-component Normal mixture; each component has cubic-in-x mean and log-sd linear in x; mixing weight = sigmoid(linear in x). 14 constants.
Validation: 5-fold CV (fit by L-BFGS, several starts). Mean NLL: cubic/lin-sd/lin-w 0.229; quartic mean 0.228 (no real gain); lin-w->quadratic 0.233; quadratic means 0.52 (bad); const-w 0.239; const sd 0.335; quadratic log-sd 0.229. Chose simplest (3,1,1).
In-sample NLL 0.2135.
expected_nll: 0.23
