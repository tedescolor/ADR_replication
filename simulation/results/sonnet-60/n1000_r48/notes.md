# Notes
Data: 1 covariate x0 in [0,1]. y has a nonlinear mean in x0 and is bimodal for x0 > ~0.55 (two branches diverge).
Validation: 5-fold CV (10-fold for the single-normal baselines), refit with several restarts in each fold.
- Single normal, cubic mean, quadratic log-sd: CV NLL 0.640 (Student-t / skew-normal gave no gain).
- Two-component normal mixture, cubic means, log-sd linear in x, weight = sigmoid(quadratic): CV NLL 0.195 (best).
  Variants that were no better: higher-degree means 0.197, quadratic log-sd 0.196-0.202, constant sds 0.22-0.29, linear weight logit 0.204.
Final model.py: 2-component Mixture of Normals, 15 fitted constants plus a few structural ones (well under 80); fit on all data by ML.
Train NLL 0.179; CV NLL 0.195.
expected_nll: 0.20
