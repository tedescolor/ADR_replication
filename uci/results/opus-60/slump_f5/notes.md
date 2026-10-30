# Notes

**Data:** 84 rows, 9 covariates, y in the range 18–59 (sd 7.6). There are no missing values and the residuals are close to Gaussian.

**Validation:** Gaussian NLL on held-out folds, using 10-fold CV repeated 20 times. Features are standardised and ridge is fit inside each fold.

**Candidates (CV NLL):**
- Constant: 3.47. Linear (raw y): 2.46. Log-y linear: 2.51 (worse than raw y).
- Linear plus forward-selected quadratic terms (x3² dominant): about 1.83.
- GP with RBF kernel and linear mean: best was about 1.41, so it does not beat the quadratic.
- Full quadratic on all 9 covariates with ridge: 1.71. Dropping x7, which is 0.91 correlated with x8 and appears to be noise in the quadratic, gives 1.34.
- Quadratic on subsets: {0,2,3,5,6,8} gives 1.33 with 27 terms and is stable across lambda. {0,2,3,5,6} gives 1.38. All 8 covariates except x7 gives 1.32–1.34 with 44 terms, but it is sensitive to lambda.
- Nested CV of automatic selection:
  - Greedy variable selection inside folds: 1.68. The inner selection is noisy.
  - Greedy term-wise elimination: 1.39, which is worse than keeping all terms.
  - So I keep the full quadratic on a fixed variable set.
- Adding cubic terms or Student-t errors did not help. The CV residuals have kurtosis −0.4, and t errors are no better than Normal.
- Calibration: the dof-corrected in-sample sigma underestimates the out-of-sample error. Inflating sigma by 1.2–1.4× improves CV NLL from 1.33 to 1.25. The final sigma is 0.82, which is about 1.28× the in-sample s of 0.64; the CV residual sd is 0.88.

**Final model:** a Normal distribution. Its mean is a full quadratic, ridge-fitted (λ=0.03) on standardised x0, x2, x3, x5, x6 and x8: 1 intercept, 6 linear and 21 quadratic terms. The sd is a constant 0.82. The program has 41 constants and passes the contract check.

**Expected NLL:** CV gives about 1.25. I add a little for optimism, because the variable subset and the sigma factor were chosen on the same CV.

expected_nll: 1.29
