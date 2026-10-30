# Notes

Data: 1000 rows, x0 in [0,1]. Scatter shows two branches: for x<0.5 a single noisy curve
(actually two overlapping ones), for x>0.55 one branch rises to ~2.5 and the other falls to ~-1.
Noise sd shrinks with x (~0.35 at x=0 to ~0.08 at x=1).

Validation: 5-fold CV repeated twice (10 folds), parameters re-fitted by ML (BFGS) on each
training split; reported mean held-out NLL (+- s.e. across folds ~0.014).

Tried (CV NLL):
- general 2-comp Gaussian mixture, poly means deg 3/4/5/6, log-sd poly, logistic weight:
  0.133 / 0.132 / 0.130 / 0.145 (higher degree sd/weight overfits).
- The free fit had weights ~0.5 and nearly identical sds, so a symmetric parametrization
  y ~ 0.5 N(c+h, s) + 0.5 N(c-h, s), c,h polynomials in t=x-0.5, log s polynomial:
  (5,5,1) 0.124, (4,4,1) 0.125, (4,3,1) 0.125, (6,6,1) 0.125, (3,3,1) 0.139,
  (5,5,2) 0.120; free weight 0.125; Student-t components 0.125 (no gain).
  (4,4,2) 0.122, (4,3,2) 0.121, (5,5,3) 0.123.

Final: symmetric (5,5,2): c and h quintic in t=x0-0.5, log s quadratic, equal weights;
15 fitted constants (+2 weights, +1 centering = 18 literals). Full-data train NLL 0.103,
CV NLL 0.120 +- 0.014. Chose it as best CV among close alternatives; adding a small allowance
for selection among ~15 configurations.

expected_nll: 0.125
