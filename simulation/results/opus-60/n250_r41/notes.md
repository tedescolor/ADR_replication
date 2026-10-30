# Notes

Data: y vs x0 in [0,1]. A single noisy branch for x < ~0.55 (U-shaped, sd shrinking with x),
which splits for x > ~0.6 into a tight rising upper branch and a lower, downward-curving branch.

Model family: 2-component Gaussian mixture with x-dependent parameters (polynomials in x),
fitted by maximum likelihood (BFGS, several random restarts).

Validation: repeated 5-fold CV (4 or 2 repeats, fixed seeds), refitting with restarts in each fold.
CV mean NLL (lower is better):
- single normal, cubic mean, quad log-sd: 0.629
- mixture, weight logit linear in x: 0.30-0.39
- mixture, weight logit quadratic (comp1 cubic mean, quad log-sd; comp2 quad mean, lin log-sd): 0.258 (R=4) / 0.268 (R=2)
- same with logit cubic: 0.269; comp2 cubic mean: 0.283
- comp1 log-sd linear: 0.255 (R=2)
- comp1 log-sd linear + comp2 constant sd (FINAL, 12 constants): 0.239 (R=2); train NLL 0.153
- Student-t first component (with linear logit) gave no gain.

Final: comp1 = Normal(cubic mean, exp(linear) sd); comp2 = Normal(quadratic mean, constant sd);
weight of comp2 = sigmoid(quadratic) (≈0 for x<0.4, ~0.5 near x=0.75, falls toward x=1).
Chosen as the simplest model with the best CV score. The train/CV gap is ~0.09; I expect new-data
NLL to be slightly above CV because model selection used the same folds.

expected_nll: 0.26
