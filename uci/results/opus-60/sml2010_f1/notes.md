# Notes

## Validation
5-fold CV on data.csv (fixed random fold assignment, seed 0). Every candidate is refit per fold
(including feature standardisation) and scored by mean Gaussian NLL on the held-out fold.

## What I tried (CV mean NLL)
- Linear Gaussian, all 20 raw covariates, constant sd: 2.99.
- Benchmarks, not usable under the contract: a tanh MLP on rank-gaussianised covariates
  (heteroscedastic Gaussian) got 2.60 with all features and 2.64 with only x0,x1,x2,x5,x6,x19,x3,x4.
  An additive piecewise-linear model in rank space got 2.69. So most of the gain comes from
  the right marginal transforms, plus a little from interactions.
- Additive shapes: x0 and x1 (corr 0.99) act in opposite directions and both saturate, so I used
  quadratics. x6 is convex and x19 concave. x3/x4 have a regime below about 192/194 where y drops
  by about 6, which I model with steep sigmoids plus min(x,240) linear terms because their upper
  tail is heavy. Log sd depends strongly on x5 and x6 and on the x3/x4 regime.
- Heavy-tailed covariates x7,x8,x11-x15 enter through logs (with clipping at 0 + offset), and x10 enters
  as an indicator. Each one is small on its own, but together they are worth about 0.08 in NLL.
- Linear mean + linear log-sd on these 19 transformed features: 2.784. Adding quadratics
  (x0,x1,x5,x6,x19) to the mean: 2.745. Quadratics in the log-sd added very little (2.742)
  and were unstable in a raw-quadratic version.
- The 500-AST-node limit forced pruning. Dropping i10, log x7 and log x15 everywhere, and
  s4 from the sd, gives CV 2.751 (the final model). A more aggressive pruning gave 2.763–2.80.

## Final program
Normal(mu, exp(clip(ls, -1, 4))). mu is linear in 16 transformed features plus 5 squares.
ls is linear in 15 transformed features. 51 constants, 470 AST nodes. Its parameters
were fit by maximum likelihood on all data. The contract check passes, including at the
corners of the domain. The clipping and min(x3/x4, 240) keep it finite at the domain extremes.

## Estimate
CV NLL 2.751. I add a small allowance for the model-selection optimism.
expected_nll: 2.76
