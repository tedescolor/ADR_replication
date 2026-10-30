# Notes

## Data
250 rows, x0 in [0,1]. Scatter shows one noisy curve (sine-like: ~-0.2 at 0, min ~-0.8 at 0.2,
max ~0.25 at 0.6, falling to ~-0.85 at 1) that bifurcates near x≈0.6: an upper branch rises
almost linearly to ~2.5 at x=1, carrying ~44% of points. Noise shrinks with x (sd ~0.3 at x=0,
~0.1-0.15 on the branches at x=1). Binned kurtosis ≈ -1.9 at high x confirms bimodality.

## Model family
Two-component normal mixture, fitted by direct maximum likelihood (BFGS + Nelder-Mead, multistart):
- lower mean m1 = a + b sin(c x + d) + e x
- upper mean m2 = m1 + k * s * softplus((x - x0)/s)   (smooth kink branching off the lower curve)
- log-linear sd for each component, constant mixture weight.
13 fitted parameters (~22 literals in model.py).

## Validation
5-fold CV, refitting everything per fold, repeated over several random splits (split-to-split
spread is ~0.02, so comparisons were made on the same splits).

| variant | params | train NLL | CV NLL |
|---|---|---|---|
| A: sine lower + softplus kink, log-linear sds (final) | 13 | 0.164 | 0.215 (4 splits: 0.211 0.204 0.226 0.219) |
| A with Student-t components | 14 | 0.164 | same (df -> ∞) |
| quintic-poly lower instead of sine | 14 | 0.167 | 0.240 (vs 0.233 for A on that split) |
| B: independent linear upper branch, x-dependent weight | 13 | 0.185 | 0.233 |
| D: quadratic log-sd + x-dependent weight | 15 | 0.163 | 0.223 |
| E: constant upper-branch sd | 12 | 0.183 | 0.230 |

A was best on all shared splits; extra flexibility (D) overfits, simplifications (E) underfit.
Final constants are the full-data MLE of A (in-sample NLL 0.1638, verified via model.py).

## Expected NLL
CV estimate ≈ 0.215 from models trained on 80% of the data; full-data fit should be slightly better.

expected_nll: 0.21
