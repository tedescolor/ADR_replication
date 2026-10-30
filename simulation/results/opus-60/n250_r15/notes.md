# Notes

## Data
250 rows, one covariate x0 in (0,1). Scatter: a single noisy curve for x < ~0.56
(dips to ~-0.85 at x~0.22, rises after), which then **splits into two branches**:
an upper branch rising roughly linearly to ~2.5 at x=1 and a lower branch staying near 0
until x~0.8 then dropping to ~-1 at x=1. About half the points above x~0.56 are on each branch.
Noise shrinks with x (sd ~0.4 near x=0, ~0.1-0.15 on both branches near x=1).

## Validation
10-fold CV repeated 3-5 times with different fold splits (refit by ML in each fold, warm-started
from the full-data fit); reported number is mean held-out NLL per row.

## Candidates (train NLL / CV NLL)
- single Normal, cubic mean, log-linear sd: 0.750 / 0.783
- single Normal, quartic mean, quadratic log-sd: 0.592 / 0.624
- 2-Normal mixture, polynomial means/log-sds, logistic-linear weight: 0.24-0.33 / 0.30-0.41
  (best: quartic main mean, quadratic lower mean, 0.240 / 0.303)
- quadratic weight logit: 0.300; Student-t main component: no gain (df -> large)
- weight = c * step(x > x*) (lower branch switched on at x* ~ 0.56): 0.222 / 0.287
- smooth step sigmoid(S(x-x*)) with S = 20/40/80/100: CV 0.291-0.294
- + cubic log-sd for main component: CV 0.283-0.291
- linear lower mean or constant lower sd: worse (0.32-0.39)
Differences among the best few (~0.005) are within CV noise (sd across repeats ~0.003).

## Final program
Mixture of two Normals:
- main: quartic mean, cubic log-sd;
- lower branch: quadratic mean, linear log-sd;
- weight of lower branch = sigmoid(a) * sigmoid(100 (x - 0.565)) (~0.48 above the split, ~0 below).
16 fitted constants (+1 fixed slope). Chosen because it encodes the visible bifurcation
structure with few parameters; the slope 100 avoids a numerically hard step while being
nearly as good. Train NLL 0.218, repeated 10-fold CV NLL 0.291.
Adding a small allowance for selection optimism:

expected_nll: 0.30
