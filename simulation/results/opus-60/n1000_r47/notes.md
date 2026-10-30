# Notes

## Data
1000 rows, x0 in (0,1), y continuous. A scatter plot (work/scatter.png) shows a single noisy
curve for x < ~0.5 (valley near x=0.22) that splits into two branches: an upper one rising
almost linearly to ~2.5 at x=1 and a lower one peaking near x=0.7 and falling to ~-1 at x=1.
Above x=0.75 there are about equal numbers of points on each branch. Residuals look Gaussian
(skew and excess kurtosis close to 0). The noise sd falls roughly linearly, from 0.35 at x=0
to 0.09 at x=1, and is about the same on both branches.

## Validation
5-fold CV (2 repeats) on the full pipeline: each fold refits the model by maximum likelihood
(BFGS, then Nelder-Mead, then BFGS, with several starts). The final model also got 4 repeats of 10-fold CV.

## Candidates (CV mean NLL)
- Generic 2-component Gaussian mixture, polynomial means, polynomial log-sds and logistic
  weights (27 params): CV 0.200 to 0.207. Overfits, and the second component is unidentified at low x.
- Structured versions with a shared log-polynomial sd and a logistic weight: 0.200 to 0.205.
- "Branching" model: mu(x) polynomial; the second branch is mu - gap(h), with h = max(x - x0, 0)
  and gap a polynomial in h with no constant term; sd = a + b x shared by both branches; weight w.
  - mu deg 5, gap linear: 0.2098 (a linear gap is too rigid)
  - mu deg 5, gap quadratic: 0.1906; with a cubic gap: 0.1905; mu deg 6: 0.1914
  - logistic weight in x instead of a constant one: 0.1911 (fitted w is about 0.49, so no gain)
  - log-quadratic sd instead of linear: 0.1907 (no gain)
  - mu deg 4, gap quadratic: 0.1902; the same with w fixed at 0.5: 0.1900
  - **mu deg 3, gap quadratic, w = 0.5: 0.1890** (10-fold x4: 0.1872, SE ~0.023)

## Final program
y | x ~ 0.5 N(mu, s) + 0.5 N(mu - gap, s), where
mu = cubic in t = 2x-1, gap = (g1 + g2 h) h with h = max(x - x0, 0), x0 ≈ 0.527, and
s = 0.390 - 0.304 x. It has 9 fitted constants (16 literals in total, counting the 2 and 1 in t, the 0 in
max, the 0.001 sd floor and the two 0.5 weights). Training NLL is 0.1785. I chose it because it is the simplest model and has the best
CV score; the larger variants are no better within noise.

Expected NLL on new data: about 0.188, between the 5-fold (0.189) and 10-fold (0.187) CV
scores. The final fit uses all 1000 rows.

expected_nll: 0.188
