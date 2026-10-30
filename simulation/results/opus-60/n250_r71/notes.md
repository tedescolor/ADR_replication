# Notes

## Data
250 rows, x0 in (0,1). Scatter shows two branches:
- a lower wavy curve over the whole range (~0.3 at x=0, min ~-0.75 at x~0.25, max ~0.25 at x~0.65, ~-1 at x=1),
  with noise that shrinks as x grows (sd ~0.43 at x=0, ~0.12 at x=1);
- an upper, roughly linear branch (y ~ -2.5 + 5.2x) that splits off where it crosses the lower
  curve (x~0.4) and carries ~40% of the mass above that point.

## Validation
5-fold CV repeated 3 times (different permutations, same folds for every candidate, so the
comparisons are paired). The loss is mean held-out NLL per row. Each fold refits all parameters by ML
(BFGS / Nelder-Mead, warm-started from the full-data fit plus random restarts).

## Candidates (CV NLL)
- 2-comp Gaussian mixture, lower mean poly deg 3/5/7/9, upper line, smooth sigmoid weight: 0.319 / 0.300 / 0.287 / 0.290
- lower mean a+bx+c*sin(wx+p) (sine), sigmoid weight: 0.298; constant weight: much worse (0.344 with poly5)
- sine + quadratic upper mean: 0.310; + quadratic log-sd: 0.299; + 2nd harmonic: 0.293; Student-t components: df -> inf (no gain)
- **sine + weight = step at crossing x 0.41 (sigmoid slope fixed at 30) times constant cap: 0.272**
  (free slope: 0.272; weight trend over x: 0.274; + 2nd harmonic: 0.275; constant upper sd: 0.277)
Paired diff of capped-step vs. smooth-sigmoid model: -0.025 +/- 0.010.

## Final model
Mixture of two Normals:
- lower: mean 0.3379 - 1.3108x + 0.8092 sin(5.7226x + 3.6458), sd exp(-0.8497 - 1.3004x)
- upper: mean -2.5471 + 5.2497x, sd exp(-0.6566 - 1.5467x)
- upper weight sigmoid(30(x - 0.4066)) * sigmoid(-0.345) (~0.41 above x~0.45)
14 constants. Train NLL 0.211, CV NLL 0.272 (SE of mean ~0.05 from row variability).
Final params fit on all data, so new-data NLL should be roughly the CV value; I add a little for
selection optimism.

expected_nll: 0.28
