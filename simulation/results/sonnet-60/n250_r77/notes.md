# Notes
Data: 1 covariate x0 in [0,1]. y has a smooth nonlinear mean (dip near x=0.25, rise after 0.45) and, for x>0.7,
becomes bimodal: a lower branch continues, and an upper branch (y rising to ~2.5 at x=1) appears.

Tried (5-fold CV, repeated with different splits, NLL per row):
- single Normal, quartic mean, log-linear sd: ~0.8 (poor, ignores bimodality)
- 2-component Normal mixture: lower comp quartic mean + log-linear sd; upper comp mean linear/quadratic, const sd;
  mixing weight sigmoid(linear in x). CV 0.29-0.34 across variants (differences within split noise ~0.03).
- Variants dropping the quad term of the upper mean, cubic lower mean, or constant s1 were all within noise.
- Quadratic lower mean: much worse (~0.6).

Final: mixture with upper mean linear (12 parameters, ~20 constants incl. structural ones). Fit by ML on all data (train NLL 0.223).
CV estimate ~0.32 (average over variants/splits, which includes optimism removal for the fitted-on-80% models).
expected_nll: 0.33
