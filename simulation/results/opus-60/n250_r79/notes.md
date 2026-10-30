# Notes

## Data
250 rows, x0 in [0,1]. Scatter shows a single noisy curve for x < ~0.55 (a dip to about -0.8
near x=0.2), which then **splits into two branches**: an upper one rising almost linearly to
about 2.5 at x=1 (tight, sd about 0.09) and a lower one bending down to about -0.85 (sd about 0.1-0.3).
Noise is clearly heteroscedastic: about 0.3 at low x, about 0.1 at high x.

## Candidates (5-fold CV, repeated 2-3 times with different splits, all params refit per fold)
- 2-component mixture with free polynomial means, log-sd and logistic weight (deg 3/1/1):
  CV 0.212; higher degrees were worse (0.23-0.30).
- "Split" structure: shared main curve g(x), lower branch = g(x) - a*relu(x-c)^p, constant
  weight, shared log-sd polynomial:
  - g quartic, relu^1: 0.209; relu^2: 0.200; relu^p: 0.194-0.199
  - g cubic, relu^p, log-sd quadratic (11 params): **0.193** (repeats 0.198/0.185/0.197)
  - g quintic: 0.205; log-sd linear: 0.199; separate sd per branch: 0.206
  - Student-t noise: 0.199 (fitted df about 4500, so effectively normal); skew-normal noise: 0.208
  - polynomial branch offset instead of relu^p: 0.220
- Training NLL of the chosen model is 0.153.

## Final
The cubic main curve with a relu^p split, a shared log-quadratic sd, and a constant weight
(0.455 upper / 0.545 lower). It has 11 fitted parameters (about 14 literals) and the best CV
of all candidates while being the most parsimonious. More flexible forms only add variance.

expected_nll: 0.20
