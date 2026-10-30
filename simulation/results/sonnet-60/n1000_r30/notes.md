# Notes
Data: 1 covariate x0 in [0,1]; y has a strongly nonlinear mean in x0 (dip near 0.25, rise to ~0.85) and a sd that varies with x0 (about 0.3 for x0<0.6, about 1.5 near x0=1).
Validation: 10-fold CV (two seeds), mean held-out NLL; all parameters refit within each fold.
Tried: Normal and StudentT (df fitted), SkewNormal; poly mean degree 3-7; log-sd poly degree 1-5.
- log-sd linear: 0.719. Quadratic: 0.614. Cubic: 0.612. Quartic: about 0.604.
- StudentT and SkewNormal gave no gain (residual kurtosis about -0.6, so tails are not heavy).
- mean degree 5 with log-sd degree 4: CV 0.6035/0.6062. Degree 6 gave about the same, so I kept the simpler one.
Final: Normal(mu(x), exp(poly4(x))) with a degree-5 polynomial mean. 11 fitted coefficients, plus the integer exponents counted as literals (passes the contract check).
Train NLL 0.596; CV NLL about 0.605. The CV is slightly optimistic because the degrees were chosen on it.
expected_nll: 0.61
