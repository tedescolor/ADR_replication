# Notes

## Data
1000 rows, x0 in [0,1]. A scatter plot shows a single noisy curve for x0 < ~0.5. Above that it
splits into two roughly equally populated branches: an upper one rising to ~2.5 and a lower one
falling to ~-1 at x0 = 1. The noise sd shrinks from ~0.3 at x0 = 0 to ~0.08 at x0 = 1. Residuals
within each branch look Gaussian (excess kurtosis ~0, skew ~-0.1).

## Candidates (mean NLL; 5-fold CV repeated over 2 seeds unless noted)
- Mixture of 2 normals, separate polynomial means, sigmoid gate on the upper weight, various sd
  models (exp-quadratic per branch, tied linear, etc.): train 0.134–0.148, CV 0.157–0.162.
  Free max weight did not help.
- Structured version: upper = lower + cubic in relu(x0 - c), constant weights 0.5/0.5 (so below c
  it is a single normal), shared sd:
  - lower deg 8, tied linear sd: CV 0.1538 (14 params)
  - lower deg 8, tied exp-quadratic sd: CV 0.1532 (15 params)  <- chosen
  - lower deg 6/7/9/10, a quartic difference, separate sd per branch: CV 0.1538–0.1579
- Sinusoidal lower mean (sin, sin+quadratic, sum of two sines): CV 0.1538–0.165, no gain.
- Final check with 10-fold CV x 4 repeats: exp-quadratic sd 0.1550 (SE ~0.012), linear sd 0.1568,
  exp-linear sd 0.1662.

## Final program
`Mixture([0.5,0.5], [Normal(m1,s), Normal(m2,s)])` where
- m1 is a degree-8 polynomial in t = 2*x0 - 1,
- m2 = m1 + cubic (with no constant term) in max(x0 - 0.503, 0),
- s = exp(quadratic in t).

That is 16 fitted constants, plus 0.5/0.5 and the 2.0 and 1.0 in t. The parameters are fitted by
maximum likelihood (BFGS) on all the data. Train NLL is 0.138.

## Expected NLL
The repeated-CV estimates are 0.153–0.155, and refitting on all the data should help a little.
I allowed for a small amount of selection optimism.

expected_nll: 0.155
