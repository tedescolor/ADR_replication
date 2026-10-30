# Notes

## Data
250 rows, x0 in [0,1]. Scatter shows one noisy curve (dip to about -0.9 near x=0.25, then rising)
that splits near x≈0.58 into two branches: an upper branch that keeps rising to about 2.6 at x=1, and a
lower branch that peaks around 0.1 near x=0.7 and falls to about -0.9 at x=1. About 45% of the points
past the split are on the lower branch. The noise sd falls roughly linearly, from about 0.4 at x=0
to about 0.1 at x=1. The lower branch has the same sd as the upper branch at the same x.

## Validation
10-fold CV repeated 3 times (30 fits per candidate), with each fold refit from the same
initialisation (BFGS→Powell→BFGS). The reported number is the mean held-out NLL.

## Candidates (CV NLL)
- 2-normal mixture, logistic-in-x weight, cubic/quadratic means: 0.298–0.336. The weight leaks onto
  the lower component for x<0.5.
- Weight = w_max * sigmoid(k (x - t)) (a "branch switches on" gate), cubic main mean, quadratic
  lower mean:
  - log-linear main sd, constant lower sd: 0.266 (main mean degree 4/5: 0.270/0.273; lower mean linear: 0.317)
  - Student-t main: 0.267; log-quadratic sd: 0.270
  - **shared linear sd s = a + b x for both components: 0.2619 (12 params)**
  - linear sd + ratio for the lower component: 0.2606 (13 params); linear + constant lower sd: 0.263
  - shared log-linear sd: 0.266; shared sqrt-linear sd: 0.289
  - lower mean tied to the main mean at the split point: 0.300 / 0.271; t main with shared linear sd: 0.264

## Final
Mixture of 2 Normals. Main mean is cubic in x; the lower-branch mean is quadratic. Both use
sd = max(0.3935 - 0.3094 x, 0.05). The lower-branch weight is
sigmoid(-0.158) * sigmoid(59.7 (x - 0.578)). The model has 13 numeric literals (12 fitted, plus the
0.05 sd floor). I chose it for parsimony: its CV score is within noise of the best (0.2606).
Train NLL is 0.2015 and CV NLL is 0.262. I added a small allowance for selection optimism.

expected_nll: 0.27
