# Notes

## Data
88 rows. x0, x1, x2 sit on design levels (x2 ≈ 100/200/300, x0 ≈ 15/20/30, ...), x3 runs from 15 to 100 and x4 is continuous.
y is positive and right-skewed (0.006 to 1.09). It is mainly driven by x4 (+) and x2 (−). Residual spread shrinks
sharply with x2 (within-group sd 0.125 / 0.046 / 0.013), roughly in proportion to the mean.
On the log scale the residuals are heavy-tailed on both sides (a fitted t gives df ≈ 3), and one point (y = 0.006) is an outlier.

## Validation
Repeated 8-fold CV (3–16 random repeats with fixed seeds; every fold is refit by ML). I report the mean held-out NLL per row on the original y scale.

## Tried (CV NLL, lower is better)
- LogNormal, linear in log x or raw x: −0.94 / −0.98. Adding a heteroscedastic log-scale made it worse (overfit).
- log-Student-t: −1.09 / −1.11. This family can't be expressed with adr.dist, and a 2-component LogNormal scale mixture
  only reached −1.00.
- Gamma with log link: −1.04 to −1.07. Gamma mixtures and extra scale terms gave no gain.
- Normal or Laplace or t with a free power of the mean in the scale, or with x2/x4 scale terms: −1.0 to −1.08. The extra
  scale parameters didn't pay off.
- Linear (identity-link) mean: unstable, worse.
- **Student-t on raw y, mean m = exp(linear), scale = c·m**: −1.09 (raw x). Using log(x4) instead of x4 gives **−1.116 ± 0.005**.
  The same model with Laplace scores −1.124. Dropping x0, x1 or x3 is not consistently better: the results flip between families.
- Adding x2·x4 or x4² doesn't help.

## Final
y ~ StudentT(df, m, s·m), with m = exp(a + b0 x0 + b1 x1 + b2 x2 + b3 x3 + b4 log x4), fitted by ML on all 88 rows.
The program has 8 constants. Training NLL is −1.243; CV NLL is −1.116.
I chose t over Laplace (they are within noise of each other) because polynomial tails are more robust to outliers on new data.

The 16-repeat CV for this exact model gives −1.116. Fitting on all rows helps slightly, but the test set is small and
may contain outliers like the 0.006 point. My estimate:

expected_nll: -1.10
