# Notes

## Data
824 rows, 8 noisy covariates (x1, x2, x4 have many values near 0 with small noise, so
they can be slightly negative; x7 is strongly right-skewed, 0.49..366), y > 0 (2.3..82.5).

## Validation
5-fold CV repeated 4 times (different shuffles), mean held-out NLL per row on the original
y scale. The SE across rows is about 0.04, but comparisons between models are paired on the
same folds, so differences of about 0.01 or more are meaningful.

## What was tried (CV NLL)
- Normal, mean linear in x0..x6 plus a cubic in log x7, constant sd: 3.38 (log-normal: 3.48)
- Hyperbolic age curve with an "effective binder" ratio (physics-style, 13 params): 3.42 to 3.45
- Linear plus z*log(t) interactions and squares: 3.29 (log-normal)
- Numpy MLP benchmark (20 tanh units, heteroscedastic Gaussian, L2): 3.15 to 3.16
- Full quadratic (all 36 pairwise products of standardized covariates, log x7) plus log-t cubic,
  ridge-fitted on log y, log-sd quadratic in log t:
  - Normal on y: 3.24
  - **log-normal: 3.16**
- With covariate transforms log x0, log x3: 3.137; adding log(x1+5), log(x2+5), log(x4+1): **3.111**
- Pruning the x5/x6 quadratic terms made it worse (3.133 / 3.159), so they stay.
- Ridge λ in 0.1..3 is flat (within 0.002).

## Final program
y ~ LogNormal(m, s) with
- u = centered (log x0, log(x1+5), log(x2+5), log x3, log(x4+1), x5, x6, log x7);
- m = a full quadratic in u, plus a cubic term in u7 (46 coefficients);
- log s = quadratic in u7 (3 coefficients).

The fit is ridge regression (λ=0.3 on standardized features) of log y on the full data,
followed by a maximum-likelihood fit of the log-sd. The program uses 60 constants and 500
AST nodes, so the node budget is the binding limit. The transform shifts keep every log
argument positive on the whole domain box (min x1 = -3.05, x2 = -2.03, x4 = -0.22, x7 = 0.49).
Train NLL is 3.028. CV NLL is 3.114 with seed 0 and 3.128 with seed 1 (same model with the
`maximum` clamps removed).

I chose the full quadratic because it matched or beat the flexible MLP benchmark. It
gives a valid density everywhere in the box. Expected NLL: about 3.12 from CV, plus a small
allowance for the transform and term choices being selected on the same CV.

expected_nll: 3.13
