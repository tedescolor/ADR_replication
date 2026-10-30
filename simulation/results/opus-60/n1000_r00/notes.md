# Notes

## Data
y vs x0 (x0 in [0,1]): one noisy curve for x0 < ~0.5 (dips to -0.8 at x≈0.2, rises steeply
around x≈0.45). Beyond x≈0.55 it splits into an upper arm (rising to ≈2.5 at x=1) and a lower
arm (≈0.2 then falling to ≈-1 at x=1), roughly 50/50. The noise is Gaussian-looking (residual skew
and excess kurtosis ≈0 at low x), with sd shrinking from ≈0.35 near x=0 to ≈0.07 near x=1.

## Validation
5-fold CV (fixed permutation), mean out-of-fold NLL, multiple starts per fit. Fold-level SE ≈0.025,
mostly driven by ~10 stray points that each cost 3–5 nats.

## Tried (CV NLL)
- Two-normal mixture with polynomial means (deg 5/6), quadratic log-sd and logistic weight: 0.184–0.192.
  The components swapped and crossed, giving poorly separated branches.
- Lower mean as a polynomial, upper mean = lower + polynomial in max(x−c,0): the fitted split point
  c landed at 0.25, and this did no better.
- Two-normal mixture with cubic B-spline means (uniform knots), polynomial log-sd per component and
  polynomial logit weight, initialized by assigning points to the upper and lower branches:
  - 4 knots 0.1887 (t), 6 knots 0.1887, 8 knots + penalty 0.01 0.1863, 10 knots 0.1965
  - log-sd degree 1: 0.194, degree 2: 0.189, degree 3: 0.193
  - weight degree 0: 0.1878, degree 1: 0.1887, degree 2: 0.186
  - second-difference penalty (6 knots, 0.03): 0.188
  - Student-t components: the fitted df went to infinity (no in-sample gain), so I kept the normal.
All the reasonable variants fall within about 0.005 of each other, which is well inside the noise.

## Final
2-component normal mixture with a constant weight (≈0.5). Each component mean is a cubic spline with
6 uniform interior knots, written as a truncated-power basis (relu(u) written as u+|u|, knots shared
between the two means), and each log-sd is quadratic in x. I chose it as the simplest of the
equivalent configurations. 34 constants, 461 AST nodes. In-sample NLL 0.152; 5-fold CV NLL 0.188.
The final fit uses all 1000 rows (vs 800 in CV), so it should do slightly better than the CV figure.

expected_nll: 0.186
