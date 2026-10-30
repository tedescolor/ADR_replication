# Notes

## Data
The data has 1000 rows, with x0 in [0,1]. For x < 0.5 there is a single noisy band. Around x≈0.5–0.56 the data splits into two
branches of roughly equal probability:
- a curved branch that falls to -1 at x=1;
- a straight line rising to about 2.5 at x=1.

The noise is Gaussian (standardized residuals: kurtosis ≈ 0, skew ≈ 0.16), and both branches share it. Its sd falls
roughly linearly, from about 0.36 at x=0 to about 0.09 at x=1.

## Model (13 fitted constants)
The model is a 2-component Normal mixture:
- curved branch mean: degree-6 polynomial in t = x - 0.5;
- line branch mean: linear in t;
- shared sd: 0.0904 + 0.2712 (1 - x);
- line-branch weight: 0.5 * sigmoid(-1.17 + 19.5 t), a sharp onset near x ≈ 0.56.

## Validation
I fitted every candidate by maximum likelihood (BFGS) and scored it with 5-fold CV on held-out NLL, repeated 2× for the
last round.
- First round: polynomial curved branch with log-linear sds and a logistic weight on both branches. CV was 0.18–0.21; degree 6 was
  best, and per-branch sds did not help.
- Capping the line-branch weight (so it plateaus at about 0.5) cut CV to 0.165–0.169.
- Sinusoidal mean for the curved branch: CV 0.185, worse.

Last round (2×5-fold CV):
| variant | CV NLL |
|---|---|
| deg 6, free cap | 0.1650 |
| **deg 6, cap fixed at 0.5** | **0.1649** |
| deg 7 | 0.1654 |
| deg 8 | 0.1650 |
| deg 5 | 0.1710 |
| quadratic sd | 0.1666 |

I chose degree 6 with the cap fixed at 0.5: it is the simplest model at the CV minimum. Its in-sample NLL is 0.1519.

expected_nll: 0.165
