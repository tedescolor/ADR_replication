# Notes

Data: y vs x0 in [0,1] lies on one noisy curve (0 -> -1 at x~0.25 -> rising) that splits at
x0~0.51 into two branches: an upper one continuing to ~2.5 at x=1 and a lower one that peaks
near 0 and falls to ~-0.9. Below the split the noise is larger (sd ~0.45); on the branches it's small.

Validation: 10-fold CV, repeated twice with different fold splits. Refit by BFGS MLE in each fold
(multi-start, warm-started from the full-data fit). Mean held-out NLL:
- single Gaussian, quintic mean, log-linear sd: 0.721
- 2-comp mixture with logistic-in-x weight (various poly degrees): 0.28-0.30
- weight = wmax*sigmoid(k(x-t0)) (gated lower branch), cubic main mean: 0.27-0.30
- same gate, quartic main mean: 0.229-0.235 (A4121 0.229, A4122 0.232, A4222 0.235, Student-t main 0.230)
- quintic main mean: 0.240 (worse, overfits)

Final (A4121): main = Normal(quartic mean, log-linear sd); low branch = Normal(quadratic mean,
log-linear sd); weight 0.45 * sigmoid(huge*(x-0.514)), which is effectively a step at x0=0.514.
15 fitted parameters (16 constants). Full-data train NLL 0.178, CV 0.229. I picked it because it had
the lowest CV; I add a little for selection optimism among about 15 candidates.

expected_nll: 0.24
