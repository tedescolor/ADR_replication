# Notes
Data: x0 in [0,1]; y is bimodal at larger x: a lower branch (cubic-like mean, sd ~0.3 shrinking)
and an upper branch (rising, ~1.6-2.5 at x>0.78) whose probability grows with x.
Model: 2-component Normal mixture, lower mean cubic in x, upper mean quadratic, log-sd linear in x
per component, mixing weight = sigmoid(linear in x). 13 fitted constants. Fit by ML (BFGS, 12 random starts).
Validation: 5-fold CV (seed 1) of the full fit procedure; mean held-out NLL:
lower cubic/upper linear 0.274, lower quartic/upper linear 0.265, cubic/quadratic 0.213 (chosen),
quartic/quadratic 0.221, quintic/linear 0.243, quadratic/linear 0.349.
Training NLL of final fit is 0.140 (optimistic); CV figure is the honest estimate.
expected_nll: 0.22
