# Notes
Data: 1 covariate, nonlinear mean (dip near x=0.2, rise after 0.6), strongly heteroscedastic (sd U-shaped, large near x=1).
Candidates: Normal / StudentT, polynomial mean degree 3-6, log-sd polynomial degree 0-3, fitted by ML.
Validation: 3x repeated 5-fold CV, mean held-out NLL.
Best: Normal, quartic mean, quadratic log-sd: CV NLL ~0.634 (Student-t and higher degrees no better; constant sd ~1.03).
Final: refit on all 250 rows; 8 coefficients (+ 1 literal-free structure), Normal output.
CV picked among 32 configs, so slightly optimistic; I add a small margin.

expected_nll: 0.65
