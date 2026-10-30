# Notes

## Data
17010 rows, 81 positive-ish covariates (many heavy-tailed), target y in [-0.85, 144], strongly
bimodal (mass at 0-10 and a second mode at 70-95), small additive noise visible near 0.

## Validation
Fixed random split: 12000 train / 5010 validation (seed 0). All model fitting and feature
selection decisions inside the search used training NLL only (forward selection by gradient
score on train, swaps accepted only if train NLL improved); validation was used to compare
the few structural choices (K, number of indices). Transcribed model.py re-evaluated on the
validation rows to verify the rounding of constants (5 significant digits) costs nothing.

## Tried
- Mixture of K Normals whose logits, means and log-scales are linear in a single linear index
  h = w.f(x) (f = raw covariates and log(|x|+1) transforms), h clipped to its training range.
  8 features K=3: 3.832; K=4: 3.803; 14 features K=4: 3.753.
- Two indices (NI=2): better per feature (14 feat: 3.719) but head costs 3x more constants;
  at the 80-constant budget a single index with many features wins.
- All ~160 transformed features, single index, K=4 (no budget): val 3.581 (reference).
- L1 path for selection: poor (index scale trades off with head slopes).
- Greedy forward selection by gradient score (4 at a time), then swap refinement
  (drop smallest |w|, add largest gradient score, accept if train NLL improves).
  K=4 with 52 features: 3.642; K=5 with 46 features: 3.610; further 1-2 feature swaps: 3.608.

## Final
Single linear index of 46 features (raw and log(|x|+1)), clipped; 5-component Normal mixture
with softmax weights, means and log-scales linear in the index. 80 constants.
Validation NLL 3.608 (train 3.597; per-row NLL standard error on the 5010 val rows ~0.016;
no heavy outliers, max per-row NLL 18.6). Final constants were refit on all 17010 rows with the
same feature set (all-data in-sample NLL 3.599, so no sign of overfitting with 80 parameters).
The index is clipped to its observed range so extreme covariate values in the domain box stay valid.

expected_nll: 3.61
