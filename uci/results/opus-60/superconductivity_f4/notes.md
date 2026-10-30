# Notes

Data: y is skewed and bimodal (mass near 0-10 and near 80-90), 0.8% of values <= 0 (additive noise),
so a pure LogNormal fails; a near-zero Normal component handles those.

Model family: Mixture of K LogNormal components + one Normal(0.41, 0.45) component.
Mixture logits (softmax, Normal logit fixed 0) and LogNormal mu are linear in 6 covariates;
log-sigma is linear in 3 covariates (clipped to [-8, 7] for validity on the whole box).
Fitted by maximum likelihood with analytic gradients (L-BFGS) on standardised features,
then coefficients converted back to the raw scale.

Validation: random 80/20 holdout (13600/3410), plus 2 more random splits for the final choice.
- K=2, 6 features, all params linear: val 3.831
- K=3, 6 features: val 3.775 (log-transformed features: 3.793)
- K=4, 6 features for logits/mu, 3 for sigma: val 3.740
- feature-swap search (one position at a time): replacing x40 by log(|x33|+1) -> val 3.733;
  on two other random splits 3.701, 3.704.
Features chosen by forward stepwise regression on log(y+1) then swap search:
x27, x70, x44, log(|x33|+1), x6, x78; sigma uses x27, x70, x6.

Final: refit on all 17010 rows (in-sample NLL 3.722). 78 constants, 473 AST nodes.
Mean held-out NLL over 3 splits ~3.713; allowing for split noise:

expected_nll: 3.72
