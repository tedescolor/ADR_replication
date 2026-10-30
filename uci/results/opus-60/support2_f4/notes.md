# Notes

## Data structure
y is a mixture of three regimes, all carrying ~N(0, 0.02) jitter:
- integer spikes at 0..7 (~71% of rows; 0 alone is 34%),
- a second lattice at 0.495 + 0.668 k, k = 0..9 (~12%), with weights that fall off quadratically in log after k=0,
- a continuous band on ~[1.3, 6], mostly 2-3.5 (~17%), modelled as a 3-normal mixture.
Covariates are mostly jittered binaries. They shift the class probabilities modestly
(multinomial logistic CV class log-loss: 1.976 for intercept only, 1.903 for full linear, 1.912 for rank-2).

## Model
Mixture(integer chain, B lattice, continuous). The lattices are built by a recursive binary-mixture
`chain` helper, which keeps the AST small (the 500-node limit binds harder than the 80-literal limit).
Class probabilities: softmax over 10 logits (8 integers, B lattice, continuous), either constant
(baseline) or a + B1*s1(x) + B2*s2(x) with two sparse linear scores (rank-2 gating).

## Validation
Density params (spike sd, lattice params, continuous mixture) were fit on all data (14 params; overfit negligible).
Gating was fit on 80% and evaluated with the full density on the 20% holdout.
Unconditional in-sample NLL: 0.272.

Holdout (20%, gating fit on the other 80%):
- unconditional class weights: 0.278
- rank-2 gating on 12 binary covariates (x7,x8,x10,x11,x12,x13,x14,x16,x18,x23,x24,x25): 0.256
Feature selection was done by importance from a rank-2 fit on all data (slight optimism),
and the density parameters were fit on all rows, so I add a small margin.

## Final program
rank-2 softmax gating (10 classes) + integer lattice chain + quadratic-log-weight 0.495+0.668k lattice
+ 3-normal continuous part; 76 literals, 459 AST nodes. Contract check passes.
Not tried for lack of time: covariate-dependent continuous part / lattice-B shape, continuous covariates
(x0, x5, x6, x19, x20) in the gating, rank 3.

expected_nll: 0.26
