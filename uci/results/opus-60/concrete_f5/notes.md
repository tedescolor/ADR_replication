# Notes

Data: 824 rows, 8 jittered covariates (several columns have mass near 0; x7 is
age-like, clustered at 3,7,14,28,56,90,180,365 -> used log(x7)). y in ~[2,83].

Validation: 5-fold CV repeated twice (10 folds, same fixed splits for all candidates,
so comparisons are paired); every candidate refit by L-BFGS on each training fold.

| model | description | CV NLL |
|---|---|---|
| A | linear + quadratic log-age, homoskedastic Normal | 3.401 |
| B | linear binder score x sigmoid(log age), het. sd | 3.374 |
| C | B + log water/binder | 3.348 |
| E | s0(x)+s1(x)*la + cubic la, het. log-sd | 3.344 |
| D | s0(x)+s1(x)*sigmoid(a+b*la), log-sd linear in la,x0..x4 | 3.304 |
| F | D + quadratic x0,x3 | 3.285 |
| G | D with Student-t | 3.287 |
| H | F + Student-t | 3.277 |
| I | H + age-curve offset depends on x1,x2,x3 | 3.262 |
| J | I + quadratic x4 and x0*x3 interaction | 3.253 (se ~0.028) |
| K | J + quadratic x0,x3 in s1 | 3.226 (init from full-data J fit) |
| L | J + x0 in age offset, quadratic x2 | 3.196 (init from full-data J); 3.213 with neutral init |
| MLP | 8 tanh units, het. Normal, L2 (benchmark) | 3.265 (noisy) |

Chosen: L (Student-t, df~4.5). mean = s0(x) + s1(x)*sigmoid(a(x)); s0, s1 linear in all
covariates, s0 has quadratic terms in x0,x2,x3,x4 and an x0*x3 interaction; the age-curve
offset a = c + b*log(x7) + linear in x0..x3; log-scale linear in log(x7) and x0..x4.
Train NLL 3.136. CV 3.196 when folds start from the full-data J optimum (slightly leaky
start), 3.213 from a neutral start (a cleaner estimate). Accounting for selection
among ~12 candidates, but given that the full-data fit uses 25% more data than each fold, I estimate ~3.22.
Checked over 2000 uniform points of the covariate box: always valid; scale grows (wide,
safe) in extreme corners.

expected_nll: 3.22
