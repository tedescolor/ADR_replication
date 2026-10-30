# Notes

Data: 250 rows, x0 in [0,1]. Scatter shows a single noisy curve for x<0.5 and two branches for x>0.55:
a straight line rising to ~2.5 at x=1 (sd ~0.12), and a smooth wavy curve that falls to ~-1 at x=1.

Model family: 2-component Normal mixture with an x-dependent weight.
- lower branch: mean = A sin(w x + phi) + polynomial trend, log-sd linear in x
- upper branch: linear mean, constant sd
- gate (weight of upper branch): sigmoid of a polynomial in x, or a capped sigmoid

Validation: 10-fold CV repeated 3 times with different fold seeds, using multi-start BFGS MLE in each fold.
(Fold fits were started from the full-data optimum plus random perturbations. Each fold's
parameters were still estimated only on its training rows.)

Results (mean CV NLL over the 3 repeats; spread between repeats is about 0.005-0.01):
- polynomial mean of degree 4 / 5 / 6, linear gate: 0.228 / 0.235 / 0.207-0.212; degree 7: 0.244
- sin + linear trend, linear gate: 0.224; with quadratic log-sd: 0.225; amplitude linear in x: 0.227
- sin + linear trend, linear gate, upper log-sd linear: 0.219
- sin + linear trend, quadratic gate: 0.2034; cubic gate: 0.2052; capped gate: 0.211
- sin + cubic trend, linear gate: 0.2056
- **sin + cubic trend, quadratic gate: 0.182** (train 0.122)  <- final
- sin + cubic trend, capped gate: 0.186; sin + quartic trend: 0.186; + quadratic log-sd: 0.186

Final program: 15 fitted constants, refit on all 250 rows. The quadratic gate gives the upper
line weight ~0 for x<0.4, rising to ~0.6 near x=0.85 and ~0.5 at x=1. This matches the data:
past x~0.55, about half the points fall on each branch.

Expected NLL: the CV estimate is 0.182. Because I picked the best of ~20 variants on the same
CV, I add a small allowance for selection optimism.

expected_nll: 0.19
