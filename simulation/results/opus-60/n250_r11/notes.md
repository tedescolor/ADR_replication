# Notes

## Data
250 rows, x0 in (0,1). Scatter shows one smooth curve (dips to about -0.7 near x=0.25, then rises
to about 2.5 at x=1, with noise sd shrinking from about 0.3 to about 0.1). Past x≈0.5 a second
branch splits off and descends to about -0.7 at x=1 (sd about 0.14). About 40% of the points
with x>0.5 are on the lower branch.

## Model family
Two-component Gaussian mixture with an x-dependent gate:
- main: mean = polynomial in x, log-sd = polynomial in x
- lower branch: mean = polynomial in x, constant sd
- weight of the lower branch = sigmoid(k(x-c)) * sigmoid(g)
Fitted by maximum likelihood (BFGS, several restarts).

## Validation
Repeated 10-fold CV (2 repeats, 500 held-out predictions), with a refit in every fold.
Results (CV NLL; the CV standard error is about 0.046 but the differences between models are paired):
- main deg3 mean, lower linear, log-sd linear: 0.237
- main deg3, lower quadratic, log-sd linear: 0.170
- main deg3/4, lower quadratic, log-sd quadratic: 0.161/0.160
- main deg3, lower cubic, log-sd quadratic: 0.149 (deg4 main: 0.147; lower quartic: 0.160 worse)
- adding a log-sd slope for the lower branch: worse (0.160); Student-t tails: no gain (df→∞)
- When k is free, ML pushes the gate to a hard step at a data point. Fixing k to
  20/40/80/200 gave CV 0.153/0.147/0.146/0.147, so I fixed k=80, a smooth but sharp gate
  that is more robust than a hard step.

## Final
main cubic mean, quadratic log-sd; lower branch cubic mean, constant sd; gate k=80, c=0.492,
gate cap sigmoid(-0.349)=0.41. The program has 16 literal constants. Train NLL is 0.102 and CV NLL is 0.146.
Expected NLL on new data is about the CV value plus a small margin for model-selection optimism.

expected_nll: 0.15
