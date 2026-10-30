# Notes

## Data
1000 rows, x0 in [0,1]. Scatter: for x < ~0.55 y follows a single noisy curve (dips to ~-0.85
at x~0.2); beyond that it splits into two branches of roughly equal weight (upper rises to ~2.5
at x=1, lower peaks ~0.15 at x~0.65 and falls to ~-1). Noise sd drops steadily from ~0.39 at
x=0 to ~0.08 at x=1. Residuals look Gaussian (excess kurtosis ~0).

## Model
2-component Gaussian mixture, shared sd:
- midline m(x): quartic in t = 2x-1
- branches m(x) ± h(x), with half-gap h(x) = a*softplus(k(x-c))/k (smooth hinge, c~0.62)
- sd linear in t (0.237 - 0.156 t), weight constant (0.518/0.482).
11 fitted parameters, 15 literals in model.py.

## Validation
5-fold CV, repeated with 2 seeds, per-point paired comparison of held-out NLL; MLE by BFGS with
multiple starts on each training fold.
- Free polynomial mixtures (two independent poly means, poly log-sd, logit weight): CV 0.185-0.194.
- Structured mid ± gap: mid degree 3/4/5/6 (log-quad sd): 0.1906 / 0.1809 / 0.1812 / 0.1844 -> quartic.
- sd forms: log-linear 0.1877, log-quad 0.1809, log-cubic 0.1821, linear 0.1802 (chosen, simpler).
- Separate sd per branch, x-dependent weight: no gain.
- Gap shapes (power of softplus, extra quadratic terms): all 0.1801, differences ~1e-4 with SE ~1e-3 -> kept plain softplus.
Final: CV NLL 0.1802; in-sample NLL on all data 0.168.

expected_nll: 0.181
