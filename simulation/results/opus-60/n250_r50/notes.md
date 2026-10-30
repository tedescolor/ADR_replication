# Notes

## Data
250 rows, x0 in (0,1). Scatter: for x < ~0.55 a single noisy curve (dip to ~-0.9 at x≈0.2,
rising through 0 at x≈0.5; residual sd ≈0.33 at small x, shrinking to ≈0.15 near 0.5,
residuals ~normal, Shapiro p=0.12). Beyond x≈0.55 the data split into two tight branches
(sd ≈0.1): an upper branch continuing the curve up to ~2.5 at x=1, and a lower branch
bending down to ~-0.9 at x=1, each with about half of the points.

## Model family
Two-component Gaussian mixture in z = x-0.5:
main component mean = polynomial, log-sd = polynomial; branch component mean = polynomial,
log-sd = polynomial; branch weight = cap * sigmoid(k (x - t)).

## Validation
Repeated 5-fold CV (2–3 repeats, fixed seeds), mean held-out NLL per point, MLE by BFGS
with a few random restarts per fold. CV noise between repeat sets is about ±0.01.

| variant | train NLL | CV NLL |
|---|---|---|
| deg (4,1,1,0) free k | 0.241 | 0.325 |
| deg (4,2,1,0) | 0.236 | 0.254 |
| deg (4,2,2,1) free k | 0.143 | 0.208–0.215 |
| higher degrees (5–6, sd deg 3) | ~0.14 | 0.218–0.234 |
| (4,2,2,1), k fixed 30 | 0.146 | 0.203 |
| + Student-t components | 0.146 | 0.203 (df→large, no gain) |
| + broad outlier component | 0.120 | 0.222 (degenerate spike, overfits) |
| cubic main mean | 0.146 | 0.196 |
| cubic main, shared sd, cap = 0.5, k=30 | 0.151 | 0.190 / 0.197 (R=2 / R=3) |
| same, k=20 / k=50 | 0.154 / 0.148 | 0.201 / 0.193 |
| same, linear log-sd | 0.155 | 0.198 |
| same, linear branch mean | 0.246 | 0.306 |

Most held-out loss comes from the split region (x 0.6–0.8), where a few points lie
between the branches.

## Final program
Cubic main mean, quadratic lower-branch mean, one quadratic log-sd shared by both
components, branch weight 0.5*sigmoid(50(x-0.539)). 15 literal constants (10 fitted
coefficients + fitted switch point 0.539, plus fixed k=50, cap 0.5, centre 0.5, and the 1 in 1-w). Refit on all data: train NLL 0.148. Chosen as the simplest
variant with the best CV (0.193 over 3x5-fold).

Expected NLL on new data: the CV estimate ≈0.193–0.197, plus a small margin for selection
among variants.

expected_nll: 0.20
