# Notes

Data: 250 rows, x0 in [0,1]. Scatter shows two branches: a wavy lower curve over the
whole range (sd shrinking with x, ~0.4 at x=0 to ~0.1 at x=1) and, for x > ~0.55, a
straight upper branch (y ~ -2 + 4.6x, sd ~0.16) carrying roughly 40-55% of the mass.

Model: 2-component Normal mixture.
- lower mean: a + b x + exp(c0 + c1 x) sin(w x + phi); log sd linear in x
- upper mean linear, constant sd
- weight of upper branch: sigmoid(quadratic in x)
14 fitted constants (15 literals in model.py incl. the `1 -` weight); MLE via Nelder-Mead + BFGS with restarts.

Validation: 5-fold CV repeated with 2 seeds (refit on each training fold), mean held-out NLL.
| variant | train NLL | CV NLL |
|---|---|---|
| A: linear gate, log-linear sds | 0.172 | 0.236 |
| B: A + quadratic upper mean | 0.172 | 0.256 |
| C: A + quadratic lower log sd | 0.170 | 0.239 |
| D: A with constant upper sd | 0.175 | 0.230 |
| E: A with Student-t lower | 0.172 | 0.236 (df -> large) |
| G: A with quadratic gate | 0.157 | 0.227 |
| DH: D + plateau gate wmax*sigmoid | 0.161 | 0.227 |
| **DG: D + quadratic gate (final)** | 0.160 | **0.2205** |
| P4/P5: quartic/quintic lower mean | 0.165/0.162 | 0.224/0.223 |
| S2: constant sine amplitude | 0.163 | 0.222 |

Variants within ~0.01 are inside CV noise; DG had the best CV and is simple, so it was
chosen and refit on all data. The full-data fit should do slightly better than the
fold fits, but held-out rows have sampling noise; I estimate ~0.22.

expected_nll: 0.22
