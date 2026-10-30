# Conditional density for y — notes

## Data findings
- y takes values on a 0.001 grid (51 levels, 0.950..1.000, roughly uniform) plus Gaussian
  noise with sd ~1.47e-4. That noise sd is the same in every operating point and close to normal.
- The covariates form 9 operating points (x1/3 rounds to 1..9, and x0 is tied to it). Inside
  each point, y depends almost linearly on the other (noisy) covariates. A linear model fitted
  separately per cluster with 5-fold CV has residual sd ~0.0029. Quadratic or cubic terms per
  cluster only reach ~0.0028, so the covariate noise sets a floor.

## Model (model.py)
- g = (round(x1/3) - 5)/4, the operating-point index, computed with math.floor.
- mean = degree-8 polynomial in g + sum_j (a_j + b_j g + c_j g^2) * log(x_j), for the
  8 covariates 13,4,9,12,10,3,7,5 (chosen by forward selection; all strictly positive on the domain box).
- envelope sd = exp(degree-4 polynomial in g), so the scale changes with the operating point.
- density = a "comb": a mixture of 51 Normals placed at k/1000 (k=950..1000) with a shared
  sd s. Each weight is proportional to exp(-0.5((k/1000-mu)/sig)^2). The mixture is built
  as nested Mixtures of 5, using small helper functions (no loops). mu and sig are clipped,
  and every weight gets +1e-300, so the density stays valid anywhere in the domain box.
- 61 constants, 494 AST nodes; the contract check passes.
- Fit: OLS for the mean, then the sigma polynomial and s by maximum likelihood of the comb.

## Validation (5-fold CV, same random folds for every candidate)
| model | CV NLL |
|---|---|
| Normal, heteroscedastic, log-linear mean | -4.20 |
| comb, log-linear mean, poly6(x0) intercept, const sigma(t) quad | -4.79 |
| comb, g-poly6, 8 log feats, linear-in-g coefs, sigma quad | -4.884 |
| same, sigma quartic | -4.888 |
| g-poly8, linear-in-g coefs, sigma quartic | -4.906 |
| g-poly6, quadratic-in-g coefs, sigma quartic | -4.927 |
| **g-poly8, quadratic-in-g coefs, sigma quartic (final)** | **-4.940** |
| Student-t envelope instead of Gaussian (on the poly8/linear model) | +0.011 better, but no AST room left |

The comb gains about 0.6 nats over a continuous density. The final model's in-sample NLL is
-4.943, almost the same as its CV value, so overfitting is minor.

expected_nll: -4.93
