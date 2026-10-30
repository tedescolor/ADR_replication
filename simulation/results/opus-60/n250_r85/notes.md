# Notes

Data: 250 rows, y vs one covariate x0 in [0,1]. Scatter shows a smooth curved main branch
(dips to about -0.85 near x=0.2, peaks about 0.3 near x=0.6, falls to -0.9 at x=1). Its noise
shrinks as x grows. From x≈0.62 a second, almost linear, low-noise branch rises to about 2.7
at x=1 and carries about 42% of the mass there.

Model: two-component Normal mixture.
- Main: polynomial mean of degree 4 in (x-0.5), log-sd linear in x.
- Upper: linear mean, constant sd (≈0.14).
- Weight of the upper branch: sigmoid(k(x-x0))·sigmoid(c), with a sharp switch at x≈0.624.
13 fitted constants (17 literals in total), fitted by maximum likelihood (multi-start L-BFGS).

Validation: 10-fold CV repeated twice (refit per fold), mean held-out NLL:
- poly4 / quad logsd / quad upper / linear upper logsd (16 params): full 0.192, CV 0.298
- poly5 / lin logsd / lin upper / const sd (14): CV 0.270
- **poly4 / lin logsd / lin upper / const sd (13): full 0.200, CV 0.263 (chosen)**
- same with Student-t main component: CV 0.255. On the full data, though, df→~5000 (that is,
  Normal), so the gain is fold noise. I kept the Normal.
- constant main sd (12): CV 0.331
A first 5-fold run of the 16-param model with Nelder-Mead gave CV 0.342, so the extra flexibility overfits.

The held-out loss is dominated by a few points in the main branch's tails and near the split
point. I add a small margin for the optimism of choosing the model on these same CV results.

expected_nll: 0.27
