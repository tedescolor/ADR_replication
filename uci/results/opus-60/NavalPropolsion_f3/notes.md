# Notes

## Data
- y lies in about [0.9497, 1.0004], roughly uniform (sd 0.0147). Each covariate on its own is
  almost uncorrelated with y.
- The covariates are almost collinear (r > 0.9) and are driven by one operating point, x0. x8
  and x11 look like pure noise around constants. The signal is in small deviations of the other
  covariates from their x0-driven curves, so the model needs interactions with x0.

## Validation
- Selection: one random 7000/2547 split. Final pipeline: 5-fold CV on all rows (the OLS mean,
  the ML log-scale fit and the edge parameters are all refit inside each fold).

## What I tried (residual sd, held-out)
- Linear in all covariates: 0.0073. Covariates × poly(x0) of degree 4 (81 terms): 0.0031.
  Full quadratic: 0.0036.
- Numpy MLP with 32 hidden units: about 0.0030, which is roughly the floor. Small MLPs (3–4
  units) that would fit in the budget did worse (0.0037–0.0042).
- Greedy forward selection over terms x_j·t^k (t = standardised x0, k ≤ 4) plus t^k up to 6.
  I then grouped the chosen terms into a contiguous degree range per covariate: 47 mean
  coefficients, CV sd 0.00318.
- Noise: the residual sd depends strongly on x0 (0.004 at low x0, 0.0015 at high x0).
  log sd = cubic in t, fit by ML.
- Edges: y is bounded at 0.95 and 1.0. A truncated normal (oracle) would gain about 0.07 nats,
  but it gives −inf for any point outside the bounds. Instead I used a skew-normal: two sigmoid
  gates measure how close the prediction is to each edge, and they shift the location, shrink
  the scale and set the skew (skew clipped to ±8, so it stays robust). Gain about 0.045 nats
  in CV. A 1% wide-Normal mixture component for robustness did not help (−4.4348 vs −4.4367),
  so I dropped it.

## Final program
Mean = intercept polynomial (degree 6 in t) + Σ_k t^k · (linear combination of covariates),
with 47 coefficients. Scale = exp(cubic in t). Then the skew-normal edge correction
(5 parameters). 65 constants, 493 AST nodes.

5-fold CV NLL: Gaussian only −4.391; final −4.437 (fold sd 0.015). Training NLL −4.443.
The final model is refit on all rows, so new-data performance should be close to the CV
figure, maybe slightly better.

expected_nll: -4.43
