# Notes

## Data
17010 rows, 81 covariates, y in [-0.85, 185], bimodal (large mass at 0-20, second mode at 70-100),
with a few slightly negative values, so I used only components with full support (Normal mixture).

## Validation
Random 13000/4010 train/validation split (seed 0); the final architecture was also refit on a
second random split (seed 7). Model selection used validation NLL only.

## What I tried (validation NLL)
- Reference: an MLP mixture-density net with all 81 inputs, 64 tanh units and 4 Normal components
  reached ~3.31 at best, then overfit. It is far too large for the 80-constant budget and only shows
  what is achievable.
- A 3-component Normal mixture with gates, means and log-scales linear in the inputs, using forward
  feature selection: 1 feature gave 4.05, 2 features 3.94. Too weak; the relationships are nonlinear.
- A tiny MDN (inputs -> H tanh units -> K-component Normal mixture logits/means/log-scales):
  - 6 inputs, H=3, K=3 (57 params): 3.757
  - 16 inputs, H=2, K=3 (61 params): 3.717
  - 16 inputs, H=3, K=3 (87 params, over budget): 3.652
  - **13 inputs, H=3, K=3 (78 params): 3.686 (split 1), 3.658 (split 2)**; train NLL is about the
    same as validation, so it does not overfit.
  Inputs came from forward selection for a linear fit of log(1+y): x27,70,44,40,68,29,54,46,17,19,74,33,80.

## Final program
I picked the 13-input, 3-hidden-unit, 3-component model because it was the best that fits the
budget. It was refit on all data, starting from the solution chosen on validation. Standardisation is
folded into the first-layer weights. The mixture weights are exp(linear in h), the Normal means are
linear in h, and the scales are exp(linear in h). Because tanh is bounded, every parameter stays
finite anywhere in the covariate box. It uses 78 constants and passes the contract check.
Running model.py itself on the split-1 validation rows gives 3.676; this is slightly optimistic
because those rows were in the final refit.

Estimate: mean of the two split validation scores, about 3.67, rounded up a little for the
noise of selecting on validation.

expected_nll: 3.68
