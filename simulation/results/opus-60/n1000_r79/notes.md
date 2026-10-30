# Notes

## Data
1000 rows, one covariate x0 in (0,1). The scatter shows one noisy curve for x < ~0.45
(wavy: about -0.1 at 0, -0.8 at 0.2, 0 at 0.5). Above ~0.5 it splits into two branches:
an upper, nearly linear branch rising to ~2.5 at x=1, and a lower branch that curves down to
~-0.9. The branches look roughly equally likely. The noise is heteroscedastic: sd ≈ 0.3–0.35
at small x on the main curve, ≈ 0.1–0.15 near x=1.

## Model family
A two-component Normal mixture, with t = 2*x0 - 1:
- the lower/main branch mean is a polynomial in t, and so is the upper branch mean;
- each component's log-sd is a polynomial in t;
- the upper-branch weight is w = sigmoid(c) * sigmoid(a + b t), a smooth switch-on near x≈0.38
  that rises to about 0.49.

I fitted by maximum likelihood (Chebyshev basis, L-BFGS-B then BFGS). Random restarts all
reached the same optimum.

## Validation
5-fold CV, repeated with 2 seeds; I report the mean per-row out-of-fold NLL. Results
(degrees of lower mean / upper mean / log-sd1 / log-sd2 / weight logit):
- 4/2/2/2/1 (17 params): **0.1865** (train 0.1674) <- chosen
- 5/2/2/2/1: 0.1880; 6/2/2/2/1: 0.1895; 7/...: 0.1893; 8/2/3/1: 0.193
- 3/2/2/2/1: train 0.1889, so it underfits
- Student-t components: 0.1884 (no gain; the residuals are close to Gaussian, kurtosis ≈ 0)
- upper mean cubic: 0.1885; upper mean linear: 0.2014
- log-sd1 linear: 0.1899; cubic: 0.1901; log-sd2 linear: 0.1878; constant: 0.1973
- quadratic weight logit: 0.1868

The paired differences between the top variants are about 1–2 SE (SE ≈ 0.001), so I chose
the simplest of them. The CV per-row SE of the mean itself is about 0.023.

## Final
model.py has the 4/2/2/2/1 structure refitted on all 1000 rows: 17 fitted parameters,
20 literal constants. Its train NLL is 0.1674. The CV estimate is 0.1865 using 800-row fits;
fitting on all rows should help slightly.

expected_nll: 0.185
