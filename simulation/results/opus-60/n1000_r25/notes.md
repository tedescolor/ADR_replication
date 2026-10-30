# Notes

## Data
1000 rows, x0 in [0,1]. Scatter plot: one noisy curve for x < ~0.45 (dips to -0.8 at x~0.2,
rises to ~0.2 at x~0.6), then it **splits into two branches**. The lower branch falls to about -0.9
at x=1; the upper branch rises almost linearly to about 2.5. The split is roughly 45/55. Noise looks
Gaussian (standardized residuals: skew/kurtosis near 0). It shrinks with x, from sd ~0.36 at x=0
to ~0.08 at x=1, and is about the same on both branches.

## Models tried (5-fold CV, mean held-out NLL; 2 fold seeds for the later runs)
- 2-normal mixture, independent polynomial means/log-sds/logit weights in t=2x-1:
  (5,2,2,1,1) 0.258; (7,3,2,1,2) 0.239; (9,3,3,2,2) 0.234; (9,4,3,2,3) 0.241.
- Structured: upper mean = lower mean + u*(g0+g1*u), u=max(x-x0,0) with x0 learned;
  one log-sd (quadratic in t) shared by both; constant weight:
  lower poly deg 6: 0.2283, deg 7: 0.2285, deg 8: 0.2272, deg 9: ~0.23;
  separate sd per branch 0.2304; weight linear in t 0.2272; log-sd linear 0.2305, cubic 0.2299.
- Sine-based lower branch (a+bx+A sin(wx+phi), plus variants): 0.225-0.234. Not better, and harder to optimize.
The structured models are all within about 0.003 of each other, which is inside the CV noise.

## Final
Lower mean is a degree-6 polynomial in t, plus the gap term (x0≈0.45). The shared sd is
exp(quadratic in t) and the mixture weight is constant. 14 fitted constants plus 2 for t=2x-1.
Fit by maximum likelihood (L-BFGS) on all the data. Train NLL 0.209, CV NLL 0.228.
I picked it as the most compact of the models that are tied on CV.

expected_nll: 0.228
