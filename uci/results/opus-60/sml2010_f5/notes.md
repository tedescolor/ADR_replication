# Notes

## Validation
5-fold CV (fixed permutation, seed 0) of mean NLL on the original y scale. Models are fit by
direct NLL minimisation (L-BFGS, analytic gradients) on standardised covariates and then
expanded to raw-scale constants in model.py; the emitted file reproduces the fitted in-sample NLL
exactly. The mean terms were chosen by greedy forward selection, so I also ran a **nested** CV:
greedy selection redone inside each outer training fold using inner 5-fold CV.

## What I tried (CV mean NLL)
- constant Normal: 3.98; OLS mean, constant sd: 3.00
- linear mean + linear log-sd (Normal or Student-t, t -> df huge, so Normal): 2.92
- capping standardised covariates at 4 sd (x3, x4 have spikes up to ~14 sd): 2.86
- + squares of x0..x6, x19: 2.823
- x9 is a regime switch (about 5% of rows have x9 ~ 1, where y is high); its interactions
  with x5/x6/x0/x1 and the products x5*x6 and x0*x1 give large gains
- greedy forward selection (OLS CV RMSE) over all 20 linear terms plus all squares/products
  of 17 covariates (x16..x18 are useless): RMSE 7.6 -> 3.5 at 45 terms. Hetero-Normal on the
  top k terms: k=20 2.71, 30 2.64, 40 2.624, 45 2.618 (log-sd on x0,x1,x2,x5,x6,x9,x19)
- skew-normal errors (residuals were right-skewed and light-tailed): k=40 -> **2.615**
- the 500-AST-node limit binds before the 80-constant limit, so I chose k=40 (497 nodes,
  60 constants)

## Final program
SkewNormal(loc, exp(ls), 1.86). x3 and x4 are capped at mean+4sd. loc is a sparse quadratic
polynomial (all linear terms of x0..x15, x19 plus 31 products/squares, written as grouped dot
products). log-sd is linear in x0, x1, x2, x5, x6, x9, x19.

## Estimate
Non-nested CV is 2.615 (SE about 0.016). Nested CV of the Normal version was 2.635, against
2.624 non-nested, so the selection optimism is about 0.01. The final model is fit on all rows.
That gives about 2.625. I round up a little to allow for leftover tail risk from polynomial
extrapolation.

expected_nll: 2.63
