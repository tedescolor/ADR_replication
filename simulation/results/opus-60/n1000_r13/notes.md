# Notes

## Data
1000 rows, x0 in (0,1). y follows one noisy curve for x < ~0.55 (dip to -0.8 near x=0.2,
noise sd ~0.25) and then splits into two branches: an upper near-linear branch rising to
~2.55 at x=1 and a lower branch flat near 0 then bending down to ~-1 at x=1 (noise sd ~0.1).
Near x=0 the spread is wide (two sub-bands), which the mixture also captures.

## Model
2-component Normal mixture. Each component mean is a cubic truncated-power spline in x
(knots 0.25, 0.5, 0.75; 7 coefs each), shared sd = exp(quadratic in t=2x-1), mixing weight
sigmoid(linear in t). 27 constants. Fitted by direct maximum likelihood (L-BFGS, analytic
gradient), initialised by splitting the right-hand points into upper/lower branch.

## Validation
5-fold CV repeated over 3-4 seeds (paired comparison across variants on the same folds).
- degree-6 polynomial means, separate sds: 0.184
- same, shared sd: 0.182
- splines with 2/3/5 knots, different knot positions: 0.180-0.185
- linear log-sd (0.1865) clearly worse; cubic log-sd, quadratic weight logit, constant
  weight, separate sds: no gain (differences within ~1 SE).
Chosen model: CV NLL 0.1812 (SE across folds ~0.01), train NLL 0.156.
PIT checks per x-decile: mean ~0, sd ~1, KS p-values all > 0.14 -> calibrated.

Final model refit on all 1000 rows; expected NLL on new data about the CV value plus a
small allowance.

expected_nll: 0.183
