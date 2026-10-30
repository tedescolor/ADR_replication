# Notes

## Data
250 rows, x0 in [0,1], y in [-1.45, 2.62]. Plotting/binning y against x0 shows one curved branch
(about -0.5 at x=0, down to -0.8 near x=0.25, up to about 0.4 near x=0.65) that then splits.
For x > ~0.45 a second, upper branch shows up that is roughly linear (about 1.0 at x=0.7, 2.55 at x=1)
and gets tighter as x grows. The lower branch bends back down (about -0.6 near x=0.95).
In the high-x region about 60% of the points are on the upper branch.

## Model family
A two-component Normal mixture:
- main branch: cubic mean in x, log-sd linear in x;
- upper branch: linear mean, log-sd linear in x;
- upper-branch weight = sigmoid(k (x - c)) * wmax, with k capped at 30 so the switch can't become a hard step.

## Validation
5-fold CV, repeated with 2 different shuffles (each row is held out twice), with 4 BFGS starts per fit.
Mean held-out NLL:

| variant | #params | train | CV |
|---|---|---|---|
| upper sd constant (base) | 12 | 0.226 | 0.312 |
| base, sigmoid slope capped 30/100 | 12 | 0.231/0.228 | 0.317/0.326 |
| base + quadratic upper mean | 13 | 0.221 | 0.312 |
| base + quadratic log-sd main | 13 | 0.218 | 0.320 |
| base + quartic main mean | 13 | 0.218 | 0.317 |
| **upper log-sd linear (ns2)** | 13 | 0.179 | **0.256** |
| ns2 + cubic upper log-sd | 14 | 0.176 | 0.264 |
| ns2 without wmax | 12 | 0.210 | 0.278 |
| ns2 + quadratic upper mean | 14 | 0.175 | 0.254 |
| ns2 + quadratic main log-sd | 14 | 0.180 | 0.268 |
| ns2 + constant main sd | 12 | 0.251 | 0.333 |
| ns2 + quartic / quadratic main mean | 14/12 | 0.176/0.412 | 0.267/0.583 |
| **ns2 + slope cap 30 (final)** | 13 | 0.186 | **0.256** |

The one big gain came from letting the upper branch's sd change with x. Other changes moved the CV
NLL by less than about 0.01, which is within noise (the SE of the CV mean is about 0.05; paired
differences are smaller). I picked ns2 with the slope cap. It ties for the best CV score, and the
capped sigmoid gives a smoother switch, which is less fragile for new x near the fork.

## Final program
Refit on all 250 rows: 13 fitted constants (14 literals in the file counting the 1.0 in the weights).
Train NLL 0.186. The CV NLL of 0.256 is the honest estimate. The fit on all data uses 25% more
rows than each CV fit, so it should do slightly better than that.

expected_nll: 0.26
