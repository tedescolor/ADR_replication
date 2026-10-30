# Notes

## Data
1000 rows, x0 in (0,1). The scatter shows one noisy curve (dips to about -0.8 near x=0.22 and rises
to about 0.2 near x=0.65) that splits into two branches for x > ~0.4. The upper branch rises
almost linearly to about 2.5 at x=1, and the lower branch falls to about -0.9. About 47% of points
are on the upper branch, and this fraction looks constant in x. The noise shrinks with x.

## Model family
Two-component Gaussian mixture with a constant weight w and a shared log-quadratic sd:
  lower mean m1(x); upper mean m2 = m1 + a*h + b*h^2 with h = max(x-c, 0).
Below c the two components coincide, so the "split" is built in.

## Validation
5-fold CV repeated twice (10 fits per configuration, all parameters refit inside each fold,
held-out NLL averaged over all held-out points). Results (mean NLL):
- independent quadratic upper branch with a sigmoid gate (poly6 + sd quad): 0.199
- shared mean + linear split (a*h), poly4 / poly5 lower mean: 0.214-0.220
- shared mean + quadratic split, poly4 lower mean: 0.206
- poly5 / poly6 lower mean + quadratic split + quadratic log-sd: 0.1877 / 0.1878
- **sin form m1 = A + Bx + C sin(Dx+E), quadratic split (a -> 0), quadratic log-sd: 0.1857**
- sin plus a quadratic term (one more parameter): 0.1850 (difference within noise)
- Student-t noise variant: too slow to finish in the time budget. Standardized residuals
  have excess kurtosis of about 0.27, so I don't expect a big gain from it.

## Final
I chose the sin form (sin1). Its CV score is within noise of the best, it has fewer parameters,
and it is smooth. The linear-split term fit to about 0 and was dropped, which leaves 11 constants.
Full-data training NLL is 0.1757, and the CV NLL is 0.1857.

expected_nll: 0.187
