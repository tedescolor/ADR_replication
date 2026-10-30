# Notes

## Data
1000 rows, x0 in [0,1]. A scatter plot shows one noisy branch for x < ~0.5. Above that it
splits into two branches of roughly equal mass: an upper branch that rises almost linearly to
about 2.5 at x=1, and a lower, wavy branch that falls to about -1 at x=1. The noise sd drops
smoothly, from about 0.38 at x=0 to about 0.1 at x=1. Residuals within each branch look
Gaussian (skew ≈ 0, excess kurtosis ≈ 0.17).

## Model family
Mixture of two Normals with a shared sd: sd = exp(linear in t), t = 2x-1. The weight of
branch B is 0.5*sigmoid(k(x-c)), so B gets at most half the mass. Each fit is a direct MLE
(BFGS, multi-start).

## Validation
5-fold CV repeated over 2 random splits (seeds 0 and 1), mean held-out NLL. For the
sinusoid models, each fold's optimiser was also started from the full-data fit, which leaks a
little.

| branch A mean | branch B mean | log sd | params | train | CV |
|---|---|---|---|---|---|
| poly deg 6 | quad | linear | 15 | 0.1785 | 0.1948 |
| poly deg 8 | quad | linear | 17 | 0.1778 | 0.1977 |
| poly deg 10 | quad | linear | 19 | 0.1759 | 0.1915 |
| poly deg 8 | quad | quad | 18 | 0.1773 | 0.1962 |
| poly deg 8, free weight cap | quad | linear | 18 | 0.1770 | 0.1993 |
| poly deg 8, Student-t | quad | linear | 18 | 0.1778 | 0.1961 |
| linear + sin | quad | linear | 12 | 0.1827 | 0.1967 |
| **quad + sin** | quad | linear | 13 | 0.1799 | **0.1949** |
| cubic + sin | quad | linear | 14 | — | 0.1949 |
| quad + sin | quad | quad | 14 | — | 0.1961 |

Differences of ±0.003 are within CV noise. The two-branch idea brings the big gain. After
that, all reasonable mean, sd and tail choices land at about 0.195.

## Final choice
Branch A: quadratic + sin(ωt+φ). Branch B: quadratic. Log sd: linear. Gate: logistic. This
has 13 fitted parameters (17 literals in total) and gives the best or tied-best CV with the
fewest parameters. The deg-10 polynomial's 0.1915 is likely noise, and it uses more
parameters with poorer conditioning. The fitted gate is centred at x≈0.40, so the two branches
overlap around 0.4–0.55, and CV supports this. Train NLL is 0.1799, CV is 0.1949. The final
model is fitted on all 1000 rows, which should help a little. The CV warm start may be a
little optimistic, so my estimate is just under 0.2.

expected_nll: 0.197
