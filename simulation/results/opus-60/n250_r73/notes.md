# Conditional density estimation notes

## Data
250 rows, x0 in [0,1]. The scatter plot shows one noisy curve (dips to about -0.8 at x≈0.25,
then rises) that splits into two branches after x≈0.5–0.6. The upper branch climbs to about 2.4
at x=1 and the lower one falls to about -0.9. Each branch holds about half the points. The noise
sd shrinks with x: about 0.32 on the trunk near x=0 and about 0.11 on the branches near x=1.

## Model family
A two-component Gaussian mixture with constant weights:
- the trunk mean m(x) is a cubic polynomial;
- with r = max(x - c, 0), the branch means are m + a1*r + q1*r^2 and m + a2*r + q2*r^2
  (before the split point both components coincide);
- both components share a log-sd, which varies with x as either a quadratic, a linear function or
  a smooth step;
- variants: trunk of degree 4 or 5, Student-t components, weights depending on r.

## Validation
5-fold CV repeated 3 times (2 for the last runs); each fold is refit by maximum likelihood with
Nelder-Mead followed by BFGS. Mean held-out NLL:

| variant | #params | train NLL | CV NLL |
|---|---|---|---|
| base (cubic, linear branches, quadratic log-sd) | 11 | 0.157 | 0.228 |
| linear log-sd | 10 | 0.166 | 0.217 |
| smooth-step log-sd | 12 | 0.158 | 0.244 |
| degree-4 trunk | 12 | 0.157 | 0.230 |
| degree-5 trunk | 13 | 0.156 | 0.233 |
| quadratic branches | 13 | 0.136 | 0.216 |
| Student-t | 12 | 0.157 | 0.230 |
| x-dependent weight | 12 | 0.157 | 0.230 |
| step log-sd + t | 13 | 0.148 | 0.249 |
| step log-sd + quadratic branches | 14 | 0.136 | 0.231 |
| **quadratic branches + linear log-sd** | **12** | **0.144** | **0.202** |
| same with a degree-4 trunk | 13 | 0.144 | 0.207 |

## Final model
Quadratic branches with a linear log-sd: 12 parameters fitted on all the data, 16 literals in
model.py. It has the lowest CV NLL and is no more complex than the alternatives. Adding
Student-t tails, x-dependent weights or a higher-degree trunk did not help out of sample.

The CV estimate is 0.202. The final model is fit on 25% more data, which should help a little,
but the CV noise is about ±0.02. My estimate:

expected_nll: 0.205
