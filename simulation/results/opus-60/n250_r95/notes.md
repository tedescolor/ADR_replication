# Notes

## Data
250 rows, one covariate x0 in (0,1). Scatter plot: a single noisy curve for x < ~0.55
(dips to about -0.8 at x≈0.22, sd about 0.3–0.4), which then **splits into two branches**:
an upper branch rising almost linearly to about 2.5 at x=1 (small noise), and a lower branch
staying near 0.1 and then dropping to about -1 at x=1 (small noise). About 45% of the points
for x > 0.6 are on the lower branch.

## Model family
A two-component Normal mixture:
- main branch: polynomial mean m1(x), log-polynomial sd s1(x);
- lower branch: polynomial mean m2(x), log-polynomial sd s2(x);
- lower-branch weight w(x) = sigmoid(a + b x) * sigmoid(c) (switches on after the split).
Fitted by maximum likelihood (BFGS, several random restarts, warm start from the full-data fit).

## Validation
Repeated 5-fold CV (3 repetitions, 15 fits per configuration), refitting everything in each fold;
mean held-out NLL per point:

| m1 deg, log s1 deg, m2 deg, log s2 deg, family | #par | train | CV |
|---|---|---|---|
| 4,2,3,1 normal | 17 | 0.028 | 0.138 |
| 4,2,3,1 Student-t main | 18 | 0.028 | 0.138 |
| 4,2,3,1 skew-normal main | 18 | 0.028 | 0.143 |
| 5,2,3,1 | 18 | 0.027 | 0.142 |
| 4,2,4,1 | 18 | 0.045 | 0.157 (optimiser) |
| 4,x,2,x variants | 15–17 | 0.04–0.05 | 0.15–0.17 |
| 4,2,3,0 | 16 | 0.030 | 0.116 |
| 3,2,3,0 | 15 | 0.031 | 0.111 |
| **3,1,3,0** | **14** | **0.040** | **0.110** |
| 4,1,3,0 | 15 | 0.039 | 0.115 |
| an earlier "lower = main − a·relu(x−c)^k" structure | 14–16 | 0.065 | 0.153–0.158 |

Heavier tails / skew did not help. A constant sd on the lower branch and cubic means were
the best; the train/CV gap (~0.07) shows the variance from 14 parameters on 250 points.

## Final program
3,1,3,0: cubic main mean, log-linear main sd, cubic lower-branch mean, constant lower-branch
sd, sigmoid weight. 14 fitted constants (written in Horner form). Train NLL 0.040,
CV NLL 0.110. Since the config was chosen among ~12 by CV, I add a little for selection
optimism.

expected_nll: 0.13
