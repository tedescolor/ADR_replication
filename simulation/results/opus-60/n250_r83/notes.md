# Notes

## Data
250 rows, x0 in [0,1]. The scatter shows two branches:
- a smooth curved branch over all x (≈ -0.3 at x=0, minimum ≈ -0.9 near x=0.25, maximum ≈ 0.3 near x=0.7, ≈ -1 at x=1);
- a straight-line branch y ≈ -2.42 + 5.07x that starts near x≈0.21 (where it is at about -1.35, below
  the curve), crosses the curve near x≈0.5 and reaches ≈2.6 at x=1.
Past the start point, about half the points are on each branch. The noise sd falls from ≈0.38 at x=0 to ≈0.11 at x=1.
The same sd works for both branches.

## Validation
Each candidate was fit by maximum likelihood (BFGS, then Nelder-Mead) and scored by 10-fold CV
repeated twice (500 held-out log densities). I report the mean held-out NLL; its se is ≈0.047 (paired differences
between models are much tighter).

| model | params | train NLL | CV NLL |
|---|---|---|---|
| sine lower + line, separate sd, gate fitted near 0.6 | 13 | 0.205 | 0.265 |
| quartic lower + line, separate sd, gate near 0.52 | 13 | 0.206 | 0.258 |
| quartic, separate sd + slope, gate at 0.21 | 14 | 0.169 | 0.217 |
| quartic, shared sd, gate at 0.21, free weight | 12 | 0.166 | 0.209 |
| cubic, shared sd, free weight | 11 | 0.168 | 0.207 |
| quintic, shared sd | 13 | 0.172 | 0.211 |
| **cubic, shared sd, weight fixed 0.5, step gate** | 9–10 | 0.168 | **0.204** |
| same, gate steepness 150 / 400 | 9 | 0.173 / 0.170 | 0.213 / 0.208 |

The main improvement came from finding that the linear branch starts at x≈0.21, not ≈0.5. The early fits
had stopped in a local optimum with the gate near 0.5–0.6. A shared log-linear sd and a fixed weight of 0.5
(the free fit gave 0.502) also helped.

## Final program
2-component Normal mixture:
- branch 1: cubic mean;
- branch 2: linear mean;
- shared sd: exp(a + b·x);
- weight of branch 2: 0.5·sigmoid(2000·(x − 0.2098)), which is effectively a step.

The program has 12 literal constants. The CV NLL of this setup (fixed steepness 2000) is 0.204; the in-sample NLL is 0.168.
I picked the final variant after comparing about 12 variants, so I add a small amount for selection optimism.

expected_nll: 0.215
