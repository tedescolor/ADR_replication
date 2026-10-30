# Notes

## Data
1000 rows, x0 in [0,1]. The scatter shows one noisy curve for x < ~0.55 (it dips to
about -0.85 near x=0.2). Past that point it splits into two branches: an upper branch
rising to ~2.5 at x=1 and a lower branch falling to ~-1. Noise sd shrinks with x (~0.4 at
x=0, ~0.1 at x=1). The two branches hold roughly equal mass.

## Model family
2-component Gaussian mixture. Component means are polynomials in x, the log-sd is a
polynomial and the logit weight is a polynomial. Fitted by direct MLE with BFGS
(work/fit.py, work/fit2.py, work/final.py).

## Validation
5-fold CV with random folds, scored as mean held-out NLL per row. The first table uses
separate sds per component, seed 0 (fit.py):

| degrees (m1,m2,logsd,logit w) | params | train NLL | CV NLL |
|---|---|---|---|
| 5,5,1,0 | 17 | 0.205 | 0.223 |
| 5,5,2,0 | 19 | 0.199 | 0.220 |
| 6,6,2,0 | 21 | 0.195 | 0.219 |
| 5,5,2,1 | 20 | 0.194 | 0.217 |
| 6,6,2,1 | 22 | 0.192 | 0.217 (0.218 in the fit2 rerun) |
| 7,7,2,1 | 24 | 0.195 | 0.223 |
| 7,7,3,1 | 26 | 0.190 | 0.224 |
| 8,8,3,1 | 28 | 0.191 | 0.224 |

Variants in fit2.py, same folds:
- Student-t components: 6,6,2,1,t 0.220; 5,5,2,1,t 0.219. No gain.
- Shared sd across components ("sh"): 6,6,2,1 0.2117; 6,6,2,0 **0.2111**;
  6,6,3,1 0.2123; 6,6,2,2 0.2122; 5,5,2,1 0.2151; 6,6,1,1 0.2181.

Sharing the sd gave a clear gain of about 0.006. Among the shared-sd fits, any
differences below about 0.002 are within fold noise.

## Final
Shared-sd mixture, degree 6 means, quadratic log-sd, constant weight (6,6,2,0 sh), refit
on all the data: 14+3+1 = 18 fitted constants, train NLL 0.197, CV NLL 0.211. I picked
it as the smallest model tied for best CV. PIT residuals of the earlier fit looked
calibrated in every x decile. My expected NLL is set slightly above CV to allow for
selection optimism.

expected_nll: 0.215
