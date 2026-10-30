# Notes

## Data
800 rows, 6 covariates (x0 counts 1..51, x1 counts heavy-tailed, x2 counts, x3 binary,
x4 in (0,25), x5 positive heavy-tailed), bivariate positive response (y0 in ~[0.9,10.3],
y1 in ~[0.003,4.3]). Transforms used: log x0, log(1+x1), log(1+x2), x3, f4=(x4-14)/10 plus f4^2
(binned means show a U-shaped x4 effect), log x5; an indicator x0==1 also helped y0.

## Validation
5-fold CV (2 repeats for model comparison, 3 other fold seeds for the final joint check),
all fits by penalised MLE (tiny ridge 1e-3), held-out NLL averaged per row.
The joint NLL is the sum of NLL(y0|x) and NLL(y1|y0,x) (Sequential factorisation).

## y0 | x
- Normal 1.605, Gamma 1.538, LogNormal 1.531 (log-variance linear in all features).
- Variance only on (log x0, x3, f4): 1.525; + indicator x0==1 in the mean: 1.518.
- Skew / Student-t in log space: no gain. Other extra terms (squares, interactions): <0.002 or worse.

## y1 | y0, x
- Normal with log y0: 0.959; with y0 linear: 0.943; with both y0 and log y0: 0.938.
- Gamma 1.02, sqrt-normal 0.957, normal+gamma mixture 0.934, Student-t 0.933.
- SkewNormal (constant shape), loc linear in features + y0 + log y0, log-scale linear in
  (log x5, y0): 0.927 (best).
- A normal truncated at 0 would do better (~0.910), but no truncated family is in the
  public `adr.dist` API, so I didn't use it.

## Final program
Sequential(LogNormal(mu(x), s(x)), y0 -> SkewNormal(loc(x,y0), scale(x,y0), shape)).
27 fitted coefficients (+ a few transform constants), 32 literals in total, 322 AST nodes.
Joint 5-fold CV NLL: 2.440 / 2.446 / 2.444 over 3 fold seeds (mean 2.443); in-sample 2.406.
I add a small allowance for selection optimism.

expected_nll: 2.45
