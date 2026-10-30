# Notes

Data: 1000 rows, x0 in [0,1]. Scatter shows a single noisy "trunk" for x < ~0.5
(U-shaped mean, sd ~0.38 at x=0 shrinking to ~0.25), which splits around x ~ 0.52 into
two narrow branches of about equal weight: an upper branch rising to ~2.5 at x=1 and a lower
branch falling to ~-1 at x=1, with noise shrinking to ~0.11 at x=1.

Model family: 2-component normal mixture. Upper mean = polynomial in u = x-0.5; lower
mean = upper mean minus a polynomial in a softplus ramp r = softplus(t(x-k))/t, so the two
components coincide on the trunk and separate after the breakpoint k. Log-scales are
polynomials in u. Weights are constant. Fitted by direct maximum likelihood (L-BFGS, multiple
random starts).

Validation: 5-fold CV (random folds) of the full fitting procedure, mean held-out NLL.

Baseline (quartic mean, quadratic ramp, quadratic/linear log-scales, 15 params):
train 0.168, CV 0.179 (single 5-fold split). Cubic mean: CV 0.179; quintic: CV 0.180.

Simpler variants (shared scale for both components, fixed 0.5 weight, etc.) were worse:
CV 0.19-0.21. Letting the two components have different scales on the trunk (a scale
mixture there) clearly helps. Some of these fits also landed in poor local optima.

Stabler estimate: 3 repeats of 5-fold CV, folds warm-started from the full-data fit:
  cubic mean (3,2,2,1), 14 params: train 0.1679, CV 0.1843 (SE ~0.01)
  quartic (4,2,2,1): 0.1852; quartic + quadratic sB: 0.1852; cubic + linear sA: 0.1851.
All are within noise, so I chose the simplest best one: the cubic mean (3,2,2,1).

Final model.py: Mixture of 2 Normals; about 22 literal constants. The rounded program
reproduces the training NLL (0.1679). Warm-started CV is a little optimistic, so I estimate
slightly above 0.184.

expected_nll: 0.187
