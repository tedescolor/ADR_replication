# Notes

**Data.** 1000 rows, x0 in [0,1]. A scatter plot shows two branches. For x0 < ~0.45 they overlap
in a single noisy band (sd ~0.35). After that they split: the upper branch rises almost
linearly to ~2.5 at x0=1, and the lower one peaks near 0.2 at x0~0.65, then drops to ~-1.
Fitting a 2-component mixture inside x-bins gives weights of about 0.5 everywhere. The noise sd
shrinks steadily with x0, from ~0.35 at x0=0 to ~0.1 at x0=1.

**Candidates.** All are 2-component Normal mixtures fitted by maximum likelihood (BFGS with restarts):
- separate polynomial means of degree 3–8, using t = 2x-1 for conditioning;
- a "centre ± gap" parametrisation, where gap = relu(x-x0)*poly;
- log-sd polynomial (degree 1–2) vs sd linear in t;
- sd shared vs one per component;
- weight constant, logistic-linear, or fixed at 0.5.

**Validation.** 10-fold CV repeated 3 times with different fold splits. Per-row held-out NLLs
were averaged, so models can be compared on the same rows. Selected results (CV NLL):
- deg6/6, log-linear sd shared, logistic weight: 0.1866
- deg7/7: 0.1884, deg8/8: 0.1901 (overfit)
- deg6/6 with separate sds: 0.1884; quadratic log-sd: 0.1863
- deg6/6, sd linear in t, shared, w=0.5 fixed: **0.1843** (deg6/6 lin + free weight 0.1847, deg5/6: 0.1849, deg6/5: 0.1920)

The differences among the top models are about the size of their paired standard error
(~0.003). I picked the simplest of them: equal weights, one shared linear sd, and two
degree-6 mean polynomials. That is 16 fitted constants plus 2 for t and 2 for the weights,
20 literals in total. Training NLL is 0.167. The CV NLL (0.184, SE about 0.023 from row
variability) is my estimate for new data. A model fitted on all 1000 rows should do
marginally better than the 900-row fold fits, but some selection optimism offsets that.

expected_nll: 0.185
