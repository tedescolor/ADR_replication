# Notes

## Data
1000 rows, x0 ~ U(0,1), y continuous. In the scatter plot y vs x0 has a single noisy curve
for x < ~0.4 (falls from 0 to -0.8 near x=0.2, then rises). From x ~0.4–0.55 it splits into two
branches: an upper branch that rises almost linearly to ~2.5 at x=1, and a lower branch that
levels off near 0.1 and falls to ~-1 at x=1. The noise sd shrinks with x (about 0.4 at x=0,
about 0.1 at x=1). Above the split the points divide roughly 50/50 between the branches.

## Model family
Two-component Gaussian mixture in t = 2x-1:
- lower mean: polynomial of degree DL; upper mean: polynomial of degree DU;
- log sd: polynomial of degree DS (separate per component, or shared);
- upper-branch weight: sigmoid(polynomial of degree DW).
I fitted by maximum likelihood (BFGS and Nelder-Mead restarts) with a heuristic initial split of the branches.

## Validation
5-fold CV on random folds (some configs also with a second fold seed), reporting mean held-out NLL.
| config (DL DU DS DW) | params | train | CV |
|---|---|---|---|
| 5 1 1 1 separate sd | 14 | 0.203 | 0.216 |
| 6 2 2 2 separate sd | 19 | 0.182 | 0.202 |
| 6 2 2 3 separate sd | 20 | 0.171 | 0.191 |
| 6 2 2 3 shared quad sd | 17 | 0.173 | 0.192 |
| 6 2 1 3 shared linear sd | 16 | 0.173 | 0.188–0.191 |
| 7 2 1 3 shared linear sd | 17 | 0.172 | 0.187 (seed 0), 0.190 (2 seeds) |
| 6 2 1 2 / 6 2 1 4 / 6 2 1 5 | 15–18 | | 0.198 / 0.189 / 0.191 |
| 6 1 1 3 / 6 3 1 3 / 8 2 1 4 | | | 0.201 / 0.191 / 0.188 |
| Student-t components (6 2 2 3) | 21 | 0.171 | 0.193 (df ≈ 16.5) |
| plateau weight c*sigmoid(k(t-t0)) | 16 | 0.176 | 0.192 |

The standardized residuals (each point assigned to its more likely component) have mean about 0, sd about 1,
skew 0.12 and excess kurtosis -0.10, with no pattern across x bins. So Gaussian noise
with a log-linear sd shared by both branches fits well, and heavier tails do not help.
The quadratic upper mean and cubic weight logit matter. Further degrees change
CV by less than one standard error (about 0.01).

## Final
Config 7 2 1 3 with shared linear log-sd: 17 fitted parameters, refit on all data, 20 literals in total.
Train NLL is 0.172 and CV NLL is 0.187–0.190. Allowing a little for selection optimism:

expected_nll: 0.19
