# Notes

## Exploration
- x1..x24 are highly collinear (look like ensemble members); x0 is weakly related.
- Gaussian/Student-t linear regressions (location on member subsets, log-scale on log ensemble spread):
  10-fold CV NLL 0.595 (Normal), 0.573 (t). Residuals were strongly left-skewed.
- Key finding: y <= x24 almost always (min of x24 - y is -0.009), so y = x24 - D with D >= 0.
  x24 also sits at, or just above, max(x1..x23) (median gap 0). This fits x24 ~ max(y, other members, ...).
  When g = x24 - max(x1..x23) is clearly positive, y ~= x24 (D median 0.05); when g ~ 0, D is wide (median ~0.55).

## Model
Mixture for y, both components anchored at loc = x24 - 0.009:
- near: SkewNormal(loc, 0.04, shape -3.5) (sharp, sits just under x24);
- far: SkewNormal(loc, scale, shape -e^8) (basically a half-normal below x24), with
  log scale = a + b log(sd of x1..x24) + c (x24 - mean(x1..x23));
- far weight = sigmoid(linear in tanh(10g), tanh(40g), tanh(100g));
- a 0.2% Normal(x24-0.5, 1) safety component (costs <= 0.002 nats, protects against outliers).
Fit by maximum likelihood (BFGS, 3 restarts). 25 constants.

## Validation (10-fold CV, NLL, nats)
| model | CV |
|---|---|
| t regression | 0.573 |
| 2-comp mixture, constant weights | 0.324 |
| + weight on tanh(10g) | 0.151 |
| + tanh(40g) (A) | 0.134 |
| A + far scale on log sd (B) | 0.121 |
| B + far scale on x24-mean (C) | 0.118 (train 0.104) |
| B + tanh(100g) weight (F) | 0.118 (train 0.101) |
| 3-comp variants | 0.132-0.134 (worse) |
Final = C + F combined (train 0.0954, 0.0961 with the safety component). I ran out of time before its own CV
finished; each of the two additions helped CV on its own, and the CV-train gap for these models is about 0.015-0.017.

expected_nll: 0.12
