# Notes

## Data
17010 rows, 81 covariates, y in [-0.85, 185] (continuous, skewed, many values near 0 with
small negative noise, a second mode around 70-95). All covariates continuous.

## Validation
Single random 70/30 split (seed 0): 12000 train, 5010 validation. All fitting on train only;
the validation set was used only to compare configurations (feature elimination ranked
features by first-layer weight norm, not by validation score).

## What I tried (val mean NLL)
- Linear ridge on log(y+1.5): all-feature MSE 0.37 vs kNN 0.17 -> strong nonlinearity.
- Mixture density network (tanh hidden layer -> softmax weights, means, log-scales of a
  K-component Normal mixture in y-space), trained with Adam, manual gradients:
  - all 81 features, 8 hidden, K=3 (737 consts, reference only): 3.41
  - all 81 features, 3 hidden, K=3 (282 consts): 3.54
  - 10 greedily-chosen features, 3 hidden, K=3 (69 consts): 3.72
  - backward elimination from 81 features (warm-started retraining, drop 20% by weight norm):
    - 2 hidden, K=3, 24 features (77 consts): **3.6575** (se 0.017)  <- chosen
    - 3 hidden, K=3, 13 features: 3.724
    - 3 hidden, K=2, 17 features: run did not finish within the time budget
- Features (not capacity in the head) were the bottleneck under the 80-constant budget, so the
  final model spends most constants on input weights and uses only 2 hidden units.

## Final program
f = 24 raw covariates (standardization folded into weights); h_i = tanh(b_i + w_i . f), i=1,2;
H=[1,h1,h2]; weights = softmax(3 linear fns of H), 3 Normal components with mean linear in H and
scale = exp(linear in H) (bounded because tanh is bounded, so valid on the whole box).
77 fitted constants + 1.0 = 78 literals.

## Expected NLL
Validation NLL 3.6575 (exact, computed from model.py). Slight optimism from model selection
on the same validation split; add ~0.01.

expected_nll: 3.67
