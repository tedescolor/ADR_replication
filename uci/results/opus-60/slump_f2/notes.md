# Notes

## Exploration
- 82 rows, 9 covariates. Linear Gaussian on all covariates: LOO NLL 2.43 (constant model 3.52).
- Strong nonlinearity: a ridge full quadratic (55 terms) gets LOO 1.41; a quadratic in x0,x1,x2,x3,x4,x6 gets 1.13.
- An ARD-RBF Gaussian process was used as a reference: honest 10-fold CV (hyperparameters refit
  per fold) gives NLL 0.46, with noise sd ~0.2. Its lengthscales say that x5, x7 and x8 are irrelevant
  and that y is a smooth function of x0, x1, x2, x3, x4, x6.
- Ratio / log features from domain intuition (e.g. x3/(x0+a x1+b x2)) did not help as much as this.

## Final program
- A 3-hidden-unit tanh network on (x0,x1,x2,x3,x4,x6): m = b + sum_k v_k tanh(w_k.u + c_k),
  with standardisation folded into the weights. Normal(m, sigma) noise.
- Fitted by Levenberg-Marquardt least squares with a tiny weight penalty (1e-3), best of 20 random restarts.
  On the full data the residual sd is 0.235 (rms 0.196), so the network matches the GP noise floor.
- 26 constants (18 weights, 3 biases, 3 output weights, 1 intercept, 1 sigma).
- tanh units are bounded, so the mean stays finite and sane anywhere in the covariate box.

## Validation
- 10-fold CV of the whole fitting procedure (with restarts) was used. The fits are sometimes
  trapped in poor local minima on a fold (that fold then has errors of 2-4 units), which inflates
  the CV mean. With 8 restarts per fold the earlier 2x10-fold run gave a mean of 1.26 and a median of 0.75.
- Because of that, I also ran LOO in which each fold is warm-started from the full-data solution, which measures
  the error of the basin the final model is in. This gives LOO error sd 0.288 and max |error| 0.76.
  Normal NLL is 0.196 at sigma 0.25, 0.176 at 0.30 and 0.208 at 0.35. Student-t did not beat Normal.
  This estimate is optimistic, because the warm start leaks the full-data fit.
- The final sigma is 0.33, slightly above the LOO optimum as a hedge (warm-start LOO NLL ~0.19).
- The expected NLL sits between the optimistic warm-start LOO (~0.19) and the honest GP 10-fold CV (0.46).

expected_nll: 0.35
