# Notes

## Data
1000 rows, x0 ~ U(0,1), y continuous. Scatter shows a single noisy "trunk" for x < ~0.5
(dips to about -0.75 near x=0.2), then the data **bifurcates**: an upper branch rising
almost linearly to about 2.5 at x=1, and a lower branch peaking near 0.15 and falling to
about -0.9. The two branches carry roughly equal mass. Noise sd shrinks with x, from about
0.35 at x=0 to about 0.1 on the branches.

## Models tried (5-fold CV NLL, MLE with several random restarts)
- Free 2-component Gaussian mixture, all parameters polynomial in z=2x-1:
  degrees (4,4,3,3,3) 0.208, (5,5,...) 0.212, (6,6,4,4,4) 0.206.
- Structured "trunk + hinge" mixture: trunk polynomial t(z); branch means
  t + r*(a_k + b_k r) with r = max(x - c, 0) and c fitted; log-sd polynomial in z;
  constant weight. Variants (trunk degree 5-8, log-sd degree 2-4, branch-specific
  sd slope, x-dependent weight) all scored 0.199-0.205 CV. A branch-specific sd was
  unstable across CV seeds (0.199 / 0.211) and an x-dependent weight hurt (0.2045).
- Residual diagnostics (standardised residuals, PIT histograms per x-decile) show
  near-Gaussian residuals: overall kurtosis 0.2, skew -0.15. Heavier tails were not
  needed.

## Final
Trunk polynomial of degree 6 (7 coefs), hinge split at c=0.488, quadratic branch offsets,
shared log-sd quadratic in z, constant mixture weight (about 0.48/0.52). 20 constants (contract count).
CV NLL over three 5-fold splits: 0.1999, 0.1979, 0.1993. Train NLL: 0.179.
I picked it because it had the best mean CV across seeds and was the most stable,
while staying simple.

expected_nll: 0.20
