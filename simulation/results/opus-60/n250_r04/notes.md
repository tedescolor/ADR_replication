# Notes

Data: 250 rows, x0 in (0,1). Scatter shows a single noisy curve (dip near x=0.2, then rising)
that bifurcates at x~0.61 into an upper branch (continues rising to ~2.5) and a lower branch
(falls to ~-1), each taking about half of the points.

Model: two-component Normal mixture.
- main: mean = polynomial in x, log-sd linear in x (sd ~0.45 at x=0 down to ~0.11 at x=1);
- lower branch: quadratic mean, constant sd ~0.12;
- weight of branch = sigmoid(steep*(x-0.608)) * ~0.5 (essentially a step).

Validation: 5-fold CV repeated twice (refit by MLE on each training fold), mean held-out NLL.
Polynomial degree for main mean: cubic 0.191, quartic 0.183, quintic 0.206, higher worse.


Variants (same 2x5-fold CV, quartic main mean):
- free logistic weight (steepness, location, level): cv 0.183
- Student-t main component: cv 0.171 (df goes very large, i.e. ~Normal)
- weight fixed to a hard step 0/0.5 at x>c (c fitted, 0.605): cv 0.169  <- chosen
- separate constant sd for the main/upper branch beyond 0.6: worse (0.185-0.204)

Final: quartic mean + log-linear sd for the main curve, quadratic lower branch with constant
sd, hard step to weight 0.5 at x=0.605. 16 constants. Train NLL 0.125, CV NLL 0.169.
The CV estimate is noisy (~0.01-0.02 between variants), so I add a small margin.

expected_nll: 0.18
