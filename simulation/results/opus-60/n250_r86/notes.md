# Notes

## Data
250 rows, x0 in [0,1]. For x < ~0.5 there is a single noisy branch (sd ~0.35, curving
down to -0.7 near x=0.25 and back up to 0 near x=0.5). Above ~0.5 it splits into two
tight branches (sd ~0.1-0.15): an upper branch that is roughly linear up to 2.5 at x=1,
and a lower branch that stays near 0 and then falls to -1. Above x=0.6 about half the
points sit on each branch.

## Model family
A 2-component Normal mixture:
- main/lower component: mean is a cubic in x, log-sd is a quadratic in x;
- upper component: mean is linear in x, sd is constant;
- gate: weight of the upper component = wmax * sigmoid(k (x - c)).

## Validation
5-fold CV repeated twice (10 fits per configuration, 8 random restarts each, L-BFGS MLE).
Mean held-out NLL per configuration (main mean degree, main log-sd degree, upper mean
degree, upper log-sd degree):
- polynomial logistic gate: (3,2,1,0) 0.315; (3,2,2,0) 0.323; (3,2,2,1) 0.332;
  (4,2,2,1) 0.349; (3,1,2,1) 0.341
- wmax*sigmoid gate, unconstrained: (3,2,1,0) 0.310; (4,2,1,0) 0.335; (3,1,1,0) 0.318
- same gate with c constrained to [0.45,0.70]: (3,2,1,0) 0.313; (3,3,1,0) 0.326; (3,2,1,1) 0.316
- Student-t main component: df went to about 80 (effectively Normal), CV 0.310, so no gain.

Extra flexibility consistently hurt in CV, so I kept the simplest structure.
The unconstrained gate converged to a near-hard step at x=0.28, where the upper line passes
under the left cloud. That is an artifact; it was only 0.003 better in CV, which is within
noise. I chose the constrained, smooth gate (c=0.516, k=7.9, wmax=0.587) because it fits
what the scatter plot shows. The final model has 15 constants, and its training NLL is
0.247 (the fit on all data).

expected_nll: 0.32
