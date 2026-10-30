# Notes

**Data.** For x < ~0.45 there is a single noisy branch (sd ~0.25). Above that it splits into
two branches of roughly equal weight: one rises roughly linearly to ~2.5 at x=1, and the
other peaks near 0.2 at x≈0.6 and then falls to ~-1. The noise shrinks on both branches as x grows.

**Model family.** A two-component Gaussian mixture. Each component mean is a polynomial in z=x-0.5,
each log-sd is a polynomial, and the weight is logistic-linear. I fit by maximum likelihood
(BFGS, 3 starts, with the means initialised to the visible branches).

**Validation.** 10-fold CV repeated twice (20 refits per configuration). Mean held-out NLL:

| mean deg | log-sd deg | train | CV |
|---|---|---|---|
| 3 | 0 | 0.206 | 0.306 |
| 3 | 1 | 0.161 | 0.236 |
| 3 | 2 | 0.133 | 0.234 |
| 4 | 0 | 0.201 | 0.315 |
| 4 | 1 | 0.149 | 0.238 |
| **4** | **2** | 0.130 | **0.215** |
| 5 | 1 | 0.145 | 0.248 |
| 5 | 2 | 0.123 | 0.265 |
| 6 | 1 | 0.124 | 0.230 |
| 6 | 2 | 0.109 | 0.241 |

**Not finished.** A more structured model (a lower polynomial branch plus an upper branch that
splits off at a fitted point c, with a constant weight; 13 to 16 params) reached training NLL
0.155. Its CV runs did not finish within the time budget, so I did not use it.

**Final.** Mean degree 4, log-sd degree 2, logistic weight: 18 fitted constants (plus the
centring 0.5 and literal 1s), fitted on all data. CV NLL 0.215. Choosing the best of about 12
configurations by CV adds some optimism, and the CV spread across configurations is large
(about ±0.03).

expected_nll: 0.23
