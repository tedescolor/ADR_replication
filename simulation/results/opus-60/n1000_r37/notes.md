# Notes

## Data
1000 rows, x0 in [0,1]. Scatter plot: one noisy curve (sd ~0.35 at x=0, shrinking to ~0.1 at x=1) that
splits into two branches of about equal weight above x≈0.4. The upper branch rises to about 2.5 and the
lower one falls to about -0.9. Kurtosis near -1.9 in the high-x bins confirms the bimodality.

## Validation
5-fold CV repeated twice (10 fits per config, fixed seeds), with the mean held-out NLL per row. Each fit is
a direct MLE (BFGS, 3 starts).

## Candidates (CV NLL)
- Free 2-Gaussian mixture, polynomial means (deg 3/4/5/6), log-sd poly deg 1/2, const or logistic weight:
  0.219 / 0.2055 / 0.2055 / 0.204 (4,2) / 0.1997 (6,2). Up to 22 params.
- Shared base + linear hinge (upper = base + relu(x-c)*poly): 0.20–0.234, worse.
- Shared base polynomial f(x) + upper = f + a*relu(x-c)^2, common log-quadratic sd, constant weight.
  This came from noticing that the gap between the free-fit branches is about 10*(x-0.42)^2.
  base deg5: 0.2016; deg6: **0.1991**; deg6 with sd deg3: 0.199. A free exponent instead of 2 gave 0.2027
  (fitted p≈1.87, so 2 holds). Separate sd per branch gave 0.2052, and a sinusoidal base 0.209–0.25. All worse.

## Final
The final model is base = deg-6 polynomial, upper mean = base + 8.89*max(x-0.370,0)^2, sd = exp(quadratic in x),
weight sigmoid(-0.012) ≈ 0.5. It has 13 fitted constants (plus two structural literals, 0.0 and 1.0).
It was refit on all data (train NLL 0.1858). It is the simplest model at the best CV level.
Expected held-out NLL ≈ CV value, rounded up slightly for selection optimism.

expected_nll: 0.20
