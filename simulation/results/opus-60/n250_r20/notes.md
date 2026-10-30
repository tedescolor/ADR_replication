# Notes

## Data
250 rows, x0 in (0,1). For x0 < ~0.45 y follows one noisy curve (sd ~0.2-0.4, larger at small x).
From ~0.45 the data split into two narrow branches: an upper, near-linear branch rising to ~2.5 at x=1
(sd ~0.15) and a lower branch that peaks near 0.1 around x=0.7 and falls to ~-1 at x=1.

## Model
Two-component Normal mixture:
- lower branch: mean = quartic polynomial in z = x-0.5, sd = exp(linear in z);
- upper branch: mean = linear in z, constant sd;
- upper-branch weight = sigmoid(a) * sigmoid(k (x - c)) (a step at c ~ 0.42, weight ~0.47 after it).
Fitted by maximum likelihood (L-BFGS-B, several random restarts).

## Validation
5-fold CV (fixed permutation), refitting everything in each fold. Configs are (deg lower mean,
deg lower log-sd, deg upper mean, deg upper log-sd):
- (4,2,1,0) 0.227, (3,2,1,0) 0.240, (4,1,1,0) **0.216**, (4,2,1,1) 0.256, (4,2,2,0) 0.228, (4,0,1,0) 0.327
- (5,1,1,0) 0.221, (4,1,2,0) 0.227
- earlier 10-fold x2: (4,2,1,0) 0.226, (5,2,1,0) 0.242
Chosen: (4,1,1,0), 13 fitted parameters (15 literals counting the 0.5 centring and the 1 in 1-w), train NLL 0.162, CV NLL 0.216.
The fitted gate is very steep (k large), effectively a hard switch; that is fine for the
density since both components are always defined.

The estimate below is the CV value plus a small margin for selection optimism.

expected_nll: 0.23
