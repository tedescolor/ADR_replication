# Conditional density estimation: notes

## Data
1000 rows, x0 ~ roughly U(0,1), y continuous. Scatter: a single wavy curve (minimum ≈ -0.85
at x≈0.25, maximum ≈ 0.25 at x≈0.65, ≈ -1 at x=1). From x≈0.4–0.45 a second, nearly linear
upper branch splits off and reaches ≈2.4 at x=1. About 44% of points sit on the upper branch
once it exists. Noise is Gaussian-looking (residual skew and kurtosis ≈ 0). The noise sd falls
from ≈0.35 at x=0 to ≈0.11 at x=1 and is the same on both branches.

## Validation
5-fold CV repeated with 2–3 random fold seeds (10–15 refits). Every fold refits all parameters
by maximum likelihood (BFGS, plus Nelder–Mead polishing), warm-started from the full-data fit.
Reported numbers are mean held-out NLL. The fold-level SE is about 0.01, but comparisons
between models are paired, so differences of about 0.003 or more are meaningful.

## Models tried (CV NLL)
- 2-component Gaussian mixture, polynomial means, log-linear sds, logistic weight in x
  (slow ramp): 0.2015 at best (lower mean degree 6). Sinusoidal lower mean: 0.215 (worse).
- Weight = c * sigmoid(k(t - t0)) with free plateau c: the fit chose a near-step (k very
  large), c ≈ 0.44. CV 0.181. A shared sd for both branches with quadratic log-sd and a cubic
  upper mean gave 0.1744; with a sharp k=1000, 0.1721.
- Profiling the step location showed a flat likelihood over t0 in [-0.4, 0]. The upper
  branch simply merges into the lower curve, so the step isn't needed.
- **Final structure:** constant weight; m_up = m_lo + h*(b0 + b1 h + b2 h^2) with
  h = max(0, t - t0), so the branches coincide below t0. CV results:
  (lower deg 6, cubic delta, quad log-sd) = **0.1702** (15 params); linear delta 0.1876;
  quadratic delta 0.173; delta = b*h^p 0.1707; lower deg 7 0.1711; log-sd linear 0.1728,
  cubic 0.1713.

## Final program
t = 2*x0 - 1 (clamped to [-1,1] to guard against extrapolation). Mixture of two Normals with
a shared sd = exp(quadratic in t). Lower mean is a degree-6 polynomial in t. Upper mean is the
lower mean plus a hinge-cubic. Weight is constant (0.441). There are 15 fitted constants
(21 literals in total, counting the 2.0/1.0/0.0 helper and clamp constants; 194 AST nodes). Train NLL is 0.1536; repeated 5-fold CV
NLL is 0.1702. The full-data fit uses 25% more data than each CV fit, so it should do slightly
better than CV, but selection among ~20 variants adds a small optimistic bias. These roughly
cancel.

expected_nll: 0.171
