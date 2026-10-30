# Notes

## Data
250 rows, x0 in (0,1), y in (-1.3, 2.6). For x < ~0.55 y follows one noisy wavy curve
(down to about -0.85 at x≈0.3, back up near x≈0.6), with noise sd ≈0.4 at x≈0 shrinking to ≈0.2.
For x > ~0.56 the data split into two narrow branches: a lower one that continues the wave
(down to about -0.9 at x=1) and an upper one that rises roughly linearly from about 0.7 to 2.5
(sd ≈0.13). Each branch holds about half the points.

## Validation
5-fold CV repeated 3 times (same splits for every candidate), giving the mean held-out NLL per row.
Every fit is penalised maximum likelihood (tiny L2), using BFGS followed by Nelder-Mead. The
mixtures start from a hand split of the two branches. The final model was also checked with
10-fold CV repeated 3 times.

| model | CV NLL |
|---|---|
| single Normal, cubic mean, log-quadratic sd | 0.671 |
| 2-Normal mixture, quartic/quadratic means, logistic-linear weight | 0.273 |
| same, weight = c*sigmoid(k(x-x0)) (step-in of upper branch) | 0.246 |
| + broad outlier component | 0.270 (worse) |
| + quadratic log-sd of main branch | 0.259 (worse) |
| upper-branch mean linear | 0.253 (worse) |
| upper sd constant | 0.244 |
| quintic main mean + const upper sd | 0.237 |
| weight plateau fixed at 0.5 | 0.232 |
| main mean a0+a1 x+a2 sin(a3 x+a4), const upper sd | 0.2325 |
| **sinusoid main mean + plateau fixed 0.5 (final)** | **0.2274** (10-fold x3: 0.233, se ≈0.05) |
| final with Student-t main component | 0.235 (worse) |
| hard step (fixed k=200) | optimisation failed, 0.255 |

## Final program
Mixture of 2 Normals:
- main: mean 0.2358 - 1.0382 t + 0.7567 sin(6.182 t + 3.4279), sd exp(-0.8558 - 1.2233 t)
- upper branch: mean quadratic in t, constant sd exp(-2.053)
- upper weight 0.5*sigmoid(75.83 (t - 0.5601))

That is 13 fitted parameters (15 literals in the source). It was refit on all the data;
training NLL is 0.178. I chose it because it has the lowest CV NLL and is also the most compact.
The expected NLL on new data is taken from the CV results (0.227 to 0.233), rounded up a little
to allow for selection optimism.

expected_nll: 0.235
