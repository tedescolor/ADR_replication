# Notes

**Data.** There's 1 covariate, x0 in [0,1], and 1000 rows. A scatter plot shows two branches. The main branch covers all of x: it dips to about -0.85 near x=0.25, then rises to about 2.5 at x=1. Its noise sd falls from about 0.35 to 0.07 as x grows. Past x≈0.42 a second branch splits off and holds about half the points. It sits near 0 around x=0.6–0.8, then falls to -1 at x=1. Its sd is about 0.1–0.2. The standardized residuals inside each branch are close to Gaussian (skew about 0.1, excess kurtosis about 0.1).

**Model family.** A 2-component Normal mixture:
- Main branch: Chebyshev polynomial means in t=2x-1, and a log-polynomial sd.
- Second branch: weight is w·sigmoid(k(x-c)). The MLE drives k to a hard step at c≈0.417 with w≈0.485.
- All parameters were fit jointly by MLE (BFGS).

**Validation.** 5-fold CV with fixed folds (seed 0). Each fold was refit starting from the full-data fit. Mean held-out NLL by degree (main mean, main log-sd, second mean, second log-sd):
- (3,1,2,0): 0.180
- (4,1,2,1): 0.182
- (5,2,2,1): 0.177
- (5,2,3,1): 0.176
- (6,2,3,1): 0.1754
- (6,3,3,1): 0.1752
- (6,2,4,1): 0.1753
- (7,2,3,1): 0.177
- (8,2,3,1): 0.178
- (6,2,3,0): 0.179
- (6,1,3,1): 0.181

I also tried a constant mixture weight with the second branch as a polynomial over the whole range, so the two branches overlap for x<0.4. Its CV was 0.183–0.184, which is worse. The likelihood is flat in the split location between 0.35 and 0.55, because the branches overlap there.

**Choice.** (6,2,3,1): 19 fitted parameters, about 23 literals. It is the smallest model on the CV plateau; bigger ones are no better within noise. Its training NLL is 0.161 and its CV NLL is 0.175. The final model is fit on all data, which should help it slightly compared with the 80% fits used in CV.

expected_nll: 0.175
