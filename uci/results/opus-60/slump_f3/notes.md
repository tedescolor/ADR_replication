# Notes

## Data
82 rows, 9 covariates, y in ~[20,53]. Linear OLS leaves residual sd ~2.3 (CV NLL ~2.43).
x8 has no detectable effect; x7 enters ~linearly; strong interactions among x0..x6.

## Validation
Repeated 10-fold CV (2-20 repeats) of the *whole* fitting pipeline (including GP
hyper-parameter fitting and distillation inside each fold). Forward selection scored on the
same CV was shown to be optimistic, so selection was nested when used.

## Candidates (CV mean NLL)
- constant Normal: 3.46; linear Normal (9 vars): 2.43; best linear subset: ~2.36
- additive squares: 2.12; OLS full quadratic: 2.6-3.0 (overfit, 45 params / 82 rows)
- nested forward selection over squares/interactions: 1.75
- small tanh MLPs (ML sigma): 2.16+ (overconfident)
- ARD squared-exponential GP: 0.65-0.67 (noise sd ~0.21, x8 lengthscale -> inf) - not
  expressible within the 80-constant budget.
- **GP distilled into a quadratic in x0..x7**: fit GP on train, evaluate its posterior mean
  at 3000-6000 augmented points (convex combos of train rows + N(0, 0.1) noise in
  standardized units), least-squares fit of the 45-term quadratic to it. Held-out RMS 0.56;
  Student-t(df 6-8, scale 1.3 x train RMS): CV NLL ~0.85. Normal: ~0.88-0.92.
- Adding greedily chosen cubic terms: test RMS worse (0.60), not used.

## Final program
z_j = (x_j - mean_j)/sd_j for j=0..7, m = full quadratic in z (45 coefs, distilled from the
GP fitted on all 82 rows), y ~ StudentT(df=6, m, 1.3*0.3795). 63 constants, 490 AST nodes.
In-sample NLL ~0.6; CV estimate of the pipeline ~0.85; quadratic extrapolation outside
the data cloud is a risk, so I add a small margin.

expected_nll: 0.9
