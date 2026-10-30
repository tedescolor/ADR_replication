# Notes

## Data
250 rows, x0 in (0,1). A single noisy curve for x < ~0.6 (dip to -0.9 near x=0.2,
then rising). Above ~0.6 the data split into two branches: an upper branch rising to ~2.5
at x=1 and a lower branch drifting from ~0.1 down to ~-1. About half the points for
x > 0.8 lie on each branch. The noise gets smaller as x grows: sd ~0.35 near x=0,
~0.12 on the branches.

## Model family
A two-component normal mixture, with t = 2*x0 - 1:
- A: mean is a polynomial in t, log-sd is linear in t;
- B: mean is a polynomial in t, log-sd is constant;
- weight of B = sigmoid(k*(x0 - c)).
Fitted by maximum likelihood (BFGS, then Nelder-Mead, then BFGS again, from two starting
values for the sigmoid slope).

## Validation
Repeated 5-fold CV (3 repeats, the same folds for every config), using mean held-out NLL.
Results as (degA, degB, degSA, degSB, weight form, family):
- (5,3,1,0,sig,N) 0.254;  (5,2,1,0,sig,N) 0.247;  (5,5,1,1,sig,N) 0.257
- (4,3,1,0) 0.250;  (4,2,1,0) 0.238;  **(3,2,1,0) 0.234**;  (2,2,1,0) 0.377
- constant sd for A: 0.29–0.31 (worse); quadratic log-sd: 0.240
- Student-t components: no gain (fitted df ~25);  weight p*sigmoid: 0.239–0.241
- a constant weight, or B having its own sd slope: worse
The standard error of a single CV mean is ~0.053, but paired differences between configs
are much smaller.

## Final
Config (3,2,1,0,sig,N): 12 fitted parameters, 15 literals in total. Training NLL is 0.172
and the 5-fold CV NLL is 0.234. The CV models were fitted on 80% of the data, so the
estimate is slightly pessimistic; I took ~0.23 as the expected NLL on new data.

expected_nll: 0.23
