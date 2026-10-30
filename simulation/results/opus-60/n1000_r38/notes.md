# Notes

## Data
1000 rows, x0 in [0,1]. Scatter shows a single noisy curve (wave-shaped: -0.15 at 0, min -0.8
near 0.25, rising to ~0.2 at 0.65) that **splits at x≈0.52** into two branches with ~50/50 weight:
a lower branch continuing the wave (down to -0.9 at x=1) and an upper branch rising to ~2.5.
Noise sd decreases smoothly with x (~0.33 at x=0 to ~0.10 at x=1), similar on both branches.

## Model family
Two-component Gaussian mixture:
- main/lower mean: polynomial in u=x-0.5 (degree 6-9 tried) or a sinusoid(+linear, amplitude-trend) form;
- upper mean: polynomial degree 1-3 (2 best);
- log-sd quadratic (separate or shared+offset) or linear sd;
- upper weight = p * sigmoid((x-c)/s); fitted s collapsed to ~0 (hard switch), c≈0.524, p≈0.49.
  Also tried Student-t components (df went large; no gain).

## Validation
10-fold CV (5-fold for the first batch), refitting by MLE (BFGS + Nelder-Mead) per fold.
| model | params | train NLL | CV NLL |
|---|---|---|---|
| poly7 / quad / sep log-quad sd | 19 | 0.189 | 0.207 (5f), 0.208 (10f) |
| poly7 / quad / shared log-quad sd | 17 | 0.190 | 0.209 |
| poly6 / quad / shared | 16 | 0.191 | 0.209 |
| poly7 / cubic upper | 18 | 0.189 | 0.212 |
| poly7 / linear upper | 16 | 0.206 | 0.223 |
| log-linear sd | 16 | 0.200 | 0.219 |
| Student-t | 18 | 0.191 | 0.209 |
| sin(+trend) main, linear sd | 13-14 | 0.196 | 0.209 |
| poly6/7, linear shared sd | 15-16 | 0.193 | 0.210-0.211 |

Differences among the top models are ~0.002, well within noise; chose the best-CV one
(poly7 main, quadratic upper, separate log-quadratic sds, hard switch at 0.524, weight 0.492),
~21 literals, passes the contract check.

## Estimate
CV ~0.207-0.208; allowing slight optimism from model selection:

expected_nll: 0.21
