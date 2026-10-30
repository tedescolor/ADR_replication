# Notes
Data: 1 covariate x0 in [0,1]. Exploration showed a smooth nonlinear lower branch of y plus, for x0 > ~0.65, a second well-separated upper branch (bimodal), with noise sd shrinking in x0.

Model: 2-component Normal mixture. Component 1 mean = degree-6 polynomial in x0, component 2 mean = linear in x0, log-sd linear in x0 for each, and the odds of component 2 = exp(a + b*(x0-0.7)). 17 constants. Fitted by maximum likelihood (BFGS, 4 starts).

Validation: 5-fold CV on data.csv (polynomial degrees 3..6 for the lower mean, 1..2 for the upper mean). Mean held-out NLL: deg3 0.301, deg4 0.285, deg5 0.285, deg6 0.278. Picked deg 6, refit on all data (train NLL 0.263). The choice among six candidates adds slight optimism, so I add a small margin.

expected_nll: 0.285
