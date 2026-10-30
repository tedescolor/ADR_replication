# Notes

## Data findings
- y = discrete atom + N(0, 0.02) noise, plus a continuous part. The atoms are at 0, 0.5, 1..7, plus small ones at 1.165 and 1.84 (4.526 was dropped).
  About 23% of rows are continuous, mostly in [2, 4].
- x7..x14 are a one-hot 8-level categorical, x23..x25 a 3-level one, and x11 == x17. The binaries carry ±0.02 jitter.
- x22 = 1 (3% of rows) switches y to a high regime: atoms 4..7 plus a continuous part N(5.2, 0.9).
- For the other rows, the continuous-vs-atom probability depends strongly on log(x5) (low x5 means more continuous).
  The 0.5 atom depends on x11, x4, log x5 and log(x6+15). The location of the continuous part rises with x19.

## Method
- Atom locations and the shared sd come from EM. Weights are softmax-linear in the features, and the continuous components
  are Normals with linear means. All of this was fitted by direct maximum likelihood with analytic gradients (L-BFGS).
- Validation: 5-fold CV and an 80/20 holdout (fold 0) on the non-x22 rows. The x22 group was fitted separately (5-fold CV 0.67).
- Non-x22 holdout NLL: 0.152 with no covariates, 0.021 with all covariates on all weights (~330 slopes), 0.043 with 20 pruned slopes
  (slopes only on the 0 atom, the 0.5 atom and the two continuous components). The pruning was done by magnitude.
  Fold 0 turned out easier than average: in-sample NLL of the same fits was ~0.11.
- To fit the 500-AST-node limit, I hand-simplified the pruned model: dropped the x3, x0, x10, x23 and log(x20) terms
  (intercepts shifted by feature means), dropped the 4.526 atom, and made one continuous weight constant. These edits were not refitted.
- I also tried a full flat softmax with 40 slopes, which was better (holdout 0.055 for an unconstrained pruning path), but it was too large for the node limit.

- Full 5-fold CV on the non-x22 rows, which finished afterwards: 0.209 with no covariates, 0.087 with all covariates
  (2 continuous components). This confirms fold 0 was easier than average (0.152 vs 0.209 with no covariates).
  So for the final 20-slope, hand-simplified model I expect roughly 0.11–0.13 on non-x22 rows.
  Adding 3% x22 rows at about 0.67 gives about 0.13–0.15 overall.

## Final program
Gate on x22. Main branch: Mixture of [atom 0, atom 0.5, two continuous Normals, fixed-weight block of atoms 1..7, 1.165, 1.84].
4 covariate-dependent weights, 55 literals, ~496 AST nodes.
In-sample NLL (every 3rd row): 0.123. The unconditional baseline was 0.26.

expected_nll: 0.14
