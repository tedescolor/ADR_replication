# Notes

## Data
250 rows, x0 in [0,1]. The scatter shows two crossing curves: an S-shaped branch
(0.4 at x=0, about -1.1 near x=0.23, rising to 2.6 at x=1) and a hump-shaped branch
(-0.3 at x=0, about 0.1 near x=0.68, -1 at x=1). The density is clearly bimodal for x>0.6, and the
two branches overlap at smaller x.

## Model family
I used a two-component Gaussian mixture: polynomial means of degree dm, log-polynomial
sds of degree ds, and a logistic-polynomial weight of degree dw. I fit it by maximum likelihood
(L-BFGS-B, several random restarts).

## Validation
I used 5-fold CV repeated twice (10 fits per config, each refit from scratch with restarts).
Mean held-out NLL, with se of about 0.05:

| dm,ds,dw | train | CV |
|---|---|---|
| 3,1,0 | 0.152 | 0.238 |
| **4,1,0** | 0.138 | **0.229** |
| 5,1,0 | 0.141 | 0.234 |
| 5,1,1 | 0.141 | 0.250 |
| 6,1,0 | 0.144 | 0.235 |
| 7,1,0 | 0.140 | 0.242 |
| 5,0,0 (const sd) | 0.169 | 0.253 |
| 5,2,0 | 0.119 | 0.319 |
| 6,2,1 | 0.131 | 0.281 |

Student-t components with (4,1,0) gave CV 0.234 (df=4) and 0.226 (df=8). Neither beats the
Gaussian by a meaningful margin, so I kept the Gaussian.

## Final
The final model is (4,1,0): a Gaussian mixture with quartic means, log-linear sds and a constant
weight (about 0.46/0.54). It uses 15 constants and was refit on all data (train NLL 0.138).
CV gives 0.229 with 200-row training sets. The full-data fit should do slightly better, but
selecting the model from several configs adds some optimism, so I estimate about 0.23.

expected_nll: 0.23
