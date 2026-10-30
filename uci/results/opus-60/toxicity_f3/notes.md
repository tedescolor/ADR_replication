# Notes

Data: 726 rows; x3, x4 are near-integer counts (0..6 plus small noise); y in (0.1, 9.6).
Linear-Gaussian residuals: excess kurtosis ~2.2, curvature in x0, x1, x5 and a variance
that changes with x5 (U-shaped), x0 and x1.

Validation: 10-fold CV with 2-3 repeated random splits, using the mean held-out NLL.
Fits were penalised MLE (small L2 penalty, BFGS).

| mean / log-scale | Normal | Student-t | N+wide-N mix |
|---|---|---|---|
| linear / const | 1.361 | 1.317 | 1.318 |
| linear / linear | 1.361 | 1.288 | 1.291 |
| lin+4 squares / linear | 1.366 | 1.271 | 1.274 |
| lin+4 squares / lin+4 squares | 1.278 | **1.231** | 1.233 |

The heavy tails come from the noise itself (t with df~4.5 beats Normal by ~0.05).
Log-normal and gamma families on y were much worse (1.45+). Piecewise-linear (hinge) bases
did no better than the quadratics (1.222-1.234).
I added mean interactions by greedy forward selection under CV: x0*x5, x3*x5, x2*x4, x1*x5
(the gains were ~0.016, 0.012, 0.004, 0.002). I kept the first three plus x1*x5.
Final check with 3 repeated 10-fold CV: M3/Q t = 1.1989. Adding x2*x4 gave 1.1980, which
was not worth it. Reducing the scale terms hurt (1.212-1.225).

Final program: Student-t, df 4.46, with
- mean = linear in all 6 covariates + squares of centred x0, x1, x2, x5 + x0*x5, x3*x5, x1*x5;
- log-scale = linear + the same 4 squares.
The scale is clipped to [0.3, 6]. On the data the fitted scale ranges over 0.34-5.1, but at the
corners of the domain box the quadratic would blow up. About 30 constants.
In-sample NLL is 1.159 and CV NLL is 1.199. The interactions were selected using the same CV,
so I add ~0.01 for selection optimism.

expected_nll: 1.21
