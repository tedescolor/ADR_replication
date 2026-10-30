# Conditional density model for y | x0

## Data
1000 rows, x0 in [0,1]. Scatter shows a single noisy curve for x0 < ~0.5 (a dip to about -0.8
at x0≈0.25) that splits into two branches for x0 > ~0.55. The upper branch rises to about 2.5 at x0=1,
and the lower branch peaks near 0.2 at x0≈0.65, then falls to about -0.9. Noise shrinks with x0
(sd about 0.4 at 0, 0.1 at 1). Both branches have about 50% of the mass and similar sd.

## Validation
5-fold CV, repeated with 2 seeds (10 fits per model), and multiple BFGS restarts for each fit.
The score is the mean held-out NLL over all rows.

## Candidates (train NLL / CV NLL)
- 2-Gaussian mixture with free poly means, quadratic log-sds, logistic weight: 0.233 / 0.252–0.263 (overfits)
- equal weights, poly5 / poly4 means, shared log-linear sd: 0.238 / 0.250
- same, separate sds: 0.237 / 0.252
- same, shared quadratic log-sd: 0.234 / 0.246
- **equal weights, poly5 / poly4 means, shared sd linear in x0 (P542): 0.233 / 0.2448**
- poly6/4: 0.2445; poly5/5: 0.247; poly6/5: 0.245; poly7/4: 0.245; poly5/3: 0.329
- quadratic sd: 0.247. Student-t components: df → ∞, 0.245 (no gain)
- free constant mixing weight: 0.246 (fitted weight ≈ 0.52, so 0.5 is fine)
- sine-based lower branch (a+bx+c sin(wx+p)): 0.248. Sine plus quadratic trend: 0.245.
  Hinge-shaped split for the upper branch: 0.265

## Choice
P542 (11 constants): it is the simplest model in the group that ties for best CV. Larger means do not help.
PIT histograms per x0-decile look uniform.

Final: Mixture(0.5/0.5) of Normal(m1(x), s(x)) and Normal(m2(x), s(x)). m1 is a quartic, m2 is a cubic,
and s = 0.404 - 0.311 x0. All are fitted on the full data (train NLL 0.2333).

expected_nll: 0.245
