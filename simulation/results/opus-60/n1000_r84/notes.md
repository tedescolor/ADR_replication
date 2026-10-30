# Notes

## Data
1000 rows, x0 in [0,1]. The scatter shows two curves: for x > 0.6 they split clearly into an
upper branch rising to about 2.6 and a lower branch falling to about -0.9. A free 2-component
mixture fit found both components present over the whole range of x (weight about 0.5,
roughly constant in x). For x < 0.5 the two curves overlap, which is why the left side looks
like a single wide band. Noise sd shrinks with x, from about 0.3 to 0.1. Residuals within a
component look Gaussian (kurtosis about 0, and a Student-t gave no gain).

## Validation
10-fold CV, repeated with 2 random fold seeds (20 fits per candidate), fitting by direct MLE
(BFGS). Folds were warm-started from the full-data optimum. That only sets the starting point,
and each fold is refit on its own training rows. Scores below are mean held-out NLL.

| candidate (means / sd / weight) | #par | CV NLL |
|---|---|---|
| quartic / separate log-linear sd / logistic-linear w | 16 | 0.1835 |
| quartic / shared log-linear / const w | 13 | 0.1805 |
| cubic means | 11 | 0.1897 |
| quintic means | 15 | 0.182 |
| quartic / shared log-quadratic sd | 14 | 0.1814 |
| Student-t components | 14 | 0.1806 |
| sin/cos bases (various) | 11-15 | 0.182-0.207 |
| quartic / shared linear sd | 13 | 0.1787 |
| **quartic / shared sd = s0*sqrt(1+k x) / const w** | 13 | **0.1784** |
| quartic / power sd, quadratic sd | 14 | 0.1794, 0.1787 |
| separate sqrt or linear sd | 15 | 0.181 |

## Final model
A 2-component Gaussian mixture. Each mean is a quartic in x. The two components share one sd,
sd = exp(-1.223)*sqrt(1 - 0.910 x), and have a constant weight of sigmoid(0.054) ≈ 0.51.
That is 13 fitted parameters, plus a 1 and a 0.0001 floor in the sqrt (15 literals). I chose
it because it had the best CV score and is among the most parsimonious candidates. Its
training NLL is 0.1642 and its CV NLL is 0.1784. I add a small allowance for selecting it
from about 15 candidates.

expected_nll: 0.181
