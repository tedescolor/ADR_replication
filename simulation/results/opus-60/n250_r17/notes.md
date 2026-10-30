# Notes

Data: 250 rows, x0 in [0,1]. Scatter shows a single noisy curve (dips to about -0.75 near x=0.25,
then rises) that splits into two branches past x of about 0.6: an upper branch rising almost linearly
to 2.5 at x=1, and a lower branch falling to -0.9. Noise shrinks with x (sd about 0.35 near 0, about 0.1 near 1).

Model: a 2-component Normal mixture.
- comp1: mean is a cubic in x, log-sd is linear in x (the base curve plus the upper branch)
- comp2: mean is a quadratic in x, log-sd is linear in x (the lower branch)
- weight of comp2 = sigmoid(k (x - c)) * wmax (fitted c=0.32, k=21, wmax=0.52)
14 fitted parameters, 18 literals.

Validation: MLE (BFGS, several restarts). Repeated K-fold CV, refitting everything inside each fold
(10-fold x2 and 5-fold x3).
- baseline (above): train 0.162, CV 0.231 (10-fold) / 0.224 (5-fold x3), se about 0.05
- Student-t comp1: 0.231 (no gain). Quartic/quintic comp1 mean: 0.240/0.243 (worse)
- hinge term on comp1 mean: 0.237. Log-quadratic sd1: 0.232. Cubic comp2 mean: 0.227 (about equal, more params)
- linear comp2 mean: 0.282. Constant sd2: 0.263. Constant mixture weight (overlapping branches): 0.25 to 0.30
I kept the baseline because it is the simplest model with the best CV score.

expected_nll: 0.23
