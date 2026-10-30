# Notes

## Data
250 rows, x0 in [0,1]. Scatter shows a smooth wavy "base" curve (min ~x=0.22, max ~x=0.67) with
noise sd decreasing in x (~0.3 -> ~0.12), and from x ~0.4-0.6 onward a second, tight, rising
linear branch (y ~ 5.2 x - 2.5, sd ~0.14) holding ~half of the points.

## Model (final)
Two-component Normal mixture:
- base: mean a + b x + A sin(w x + phi), sd exp(linear in x)
- upper branch: linear mean, constant sd
- weight of upper branch: sigmoid(100 (x - c)) * sigmoid(l)  (step at c ~0.41, where the line
  crosses the base curve, so the onset is not sharply identified; weight ~0.51)
12 fitted parameters (+ the fixed steepness 100 and literal 1.0), fitted by MLE.

## Validation
10-fold CV, repeated with 2 shuffles; each fold refits from scratch with multi-start
(BFGS -> Nelder-Mead -> BFGS). Mean held-out NLL:
- poly4/5/6 base + linear branch + gentle sigmoid weight (no cap): 0.223-0.29
- poly5/6 base, step weight with free cap: 0.203-0.216
- sine base, log-sd linear (FINAL): 0.1998 (train 0.142)
- sine base, log-sd quadratic: 0.1979 (within noise, one more param; not adopted)
- sine + x^2 term: 0.204; constant base sd: 0.264; upper sd linear: 0.201;
  Student-t base: 0.203; quadratic upper branch (poly base): worse.
Chose the simplest model at the CV optimum.

expected_nll: 0.205
