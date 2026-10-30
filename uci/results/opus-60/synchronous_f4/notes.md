# Notes

## Data findings
- x1 + x2 = 1 (to within ±0.004), so x1 adds nothing beyond x2.
- x0 and x2 lie on a jittered grid: x0 in steps of 0.1 from 3.0 to 6.0, x2 in steps of 0.01 from 0 to 0.35.
- **x3 is a noisy reading of y**: y = 1.1803 + 0.0009996·x3 + e, with e ~ N(0, 0.0029). The noise is
  homoscedastic and normal (Shapiro p = 0.30). In 58 of the 445 rows (13%) x3 is "missing": its value is
  ≈ 0 ± 2, in the range [-3.5, 4.9]. All other rows have x3 ≥ 36.9. I split the two groups at x3 = 18.
- When x3 is missing, y must come from f(x0, x2). This function is deterministic-looking and
  piecewise-linear with sawtooth jumps:
  - in x0 there is a sawtooth with period 0.5 and jumps at 3.05 + 0.5k;
  - in x2 there is a sawtooth with period 0.05 and jumps at 0.095 + 0.05k. It is only active for
    x2 > 0.09; below that the level is flat.
  - The slopes and amplitudes change smoothly with x0 and x2.
  - The sawtooth is written as `arctan2(-sin 2πt, -cos 2πt)/2π`.
- What remains unexplained: the x0 = 3.0 column sits above the fit, and the x2 = 0.35 row has no x0
  sawtooth. Both produce heavy-tailed residuals, so I use a Student-t.

## Model
- x3 present: Normal(1.1257 + 0.000953·x3 + 0.0465·f(x0,x2), 0.00287). Adding f gives a small
  improvement in CV (−4.420 vs −4.407).
- x3 missing: StudentT(df=3, loc=f(x0,x2), scale=0.0055).
- f is linear in 18 features: a polynomial in u = max(x2 − 0.095, 0) and z0 = x0 − 4.5, the two
  sawtooths s0 and s2, their interactions with z0 and u, and s0·s2. It is fitted by t-likelihood IRLS
  on all 445 rows. This is valid because every row's y is a draw of f + noise.
- 41 constants by the contract count (this includes structural ones such as 2, pi, 4.5 and the grid phases), 324 AST nodes.

## Validation
Repeated 10-fold CV: 5 repeats for the Normal fits, 3 repeats for the Student-t fits. f was refitted in each fold.
- Feature-set comparison: NLL of f over all held-out rows (more stable than the 58 missing rows alone):
  - Normal noise: set A (13 terms) −2.93, set C −2.98, set D −3.02, set F −2.96.
  - Student-t: set D at df = 2 / 4 / 8 gives −3.31 / −3.29 / −3.22.
  - On the missing rows only, set D with t: about −3.40 (df 2) and −3.38 (df 4). I chose df = 3.
- Present rows, combined loc: −4.42 in CV.
- Mixture: 0.87·(−4.42) + 0.13·(−3.38) ≈ −4.28. In-sample NLL of the final program: −4.31.
- I shade the estimate slightly for sampling variation in the missing fraction and for edge cells
  (x0 = 3.0, x2 = 0.35).

expected_nll: -4.26
