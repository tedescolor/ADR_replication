# Notes

## Data
250 rows, x0 in [0,1]. For x0 < ~0.62, y follows one curved branch: about -0.17 at 0, a minimum
of about -0.72 near 0.2, about 0.15 near 0.6, then down to about -0.85 at 1. Its noise shrinks
with x (sd about 0.4 near 0, about 0.15–0.2 in the middle). For x0 > ~0.63 a second, roughly
linear branch appears (y ≈ -2.3 + 4.9 x, sd ≈ 0.11) and takes about half the points.

## Model families tried (all two-Normal mixtures, fit by maximum likelihood with BFGS and multiple starts)
- Lower-branch mean: polynomial of degree 3/4/5/6, or a + b x + c sin(ωx+φ) (with and without x²).
  Degree 4 was best; degree 3 is clearly worse (CV +0.035); degree 5 and up are slightly worse.
- Lower log-sd: constant (much worse, +0.1), linear (best), quadratic (slightly worse).
- Upper branch: linear mean with constant sd (best); a log-linear sd is slightly worse.
- Gate: logistic in x (CV 0.185); a sharp step p·sigmoid(k(x-c)) with k free (k→67, CV 0.184,
  because folds pick very sharp gates and then pay heavily for upper-branch points just
  before the threshold, e.g. (0.621, 0.845)); **k fixed** at 15/25/30/40/50/70 gives CV
  0.170/0.165/0.161/0.159/0.159/0.159.
- A Student-t lower component (ν→7–8, no gain); a third wide background component (the
  free version collapses onto one outlier; with a fixed sd its weight goes to 0, no gain).

## Validation
10-fold CV repeated 2–3 times with different shuffles; each fold refits all parameters on the
training folds. The scores above are the mean held-out NLL per point.

## Final
lower = Normal(quartic(x), exp(a + b x)), upper = Normal(linear, const), with upper weight
sigmoid(p)·sigmoid(50(x - c)). The model has 12 fitted constants plus k = 50.
Training NLL is 0.105 and the 3×10-fold CV NLL is 0.159. k was chosen from the CV curve, which
adds a little optimism. A few lower-tail outliers (e.g. y = -1.59 at x = 0.27) dominate
the variance of the estimate. I estimate the NLL on new data a bit above the CV value.

Process note: one earlier exploratory fit script was still running in the background when I
started two more, so for a few minutes 3 processes ran at once.

expected_nll: 0.17
