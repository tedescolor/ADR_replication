# Notes
Data: 1 covariate x0 in [0,1]. Single Normal/StudentT with polynomial mean and log-sd gave 5-fold CV NLL ~0.632.
Binned inspection showed that for x0 > ~0.6 y splits into two clean branches (lower curve and upper curve ~+2),
with the upper branch's weight growing with x0. So the model is a 2-component Normal mixture:
- lower branch: mean = quartic in u=x0-0.5, log-sd quadratic
- upper branch: mean = cubic in u, log-sd linear
- weight of upper branch = sigmoid(quadratic in u)
17+... = 17 mean/sd/weight constants plus 0.5, 1.. (about 20 literals total), well under the 80 limit.
Validation: 5-fold CV (one fixed shuffle), 6 random restarts per fit, compared polynomial degrees
(mean1, mean2, logsd1, logsd2, weight): (3,1,1,1,1) 0.260; (4,2,2,2,2) 0.229; (4,3,1,1,2) 0.218; (4,3,2,1,2) 0.2147 (chosen); (4,4,1,1,2) 0.2215.
The chosen degrees were picked on the same CV so 0.2147 is slightly optimistic; training NLL of the final fit is 0.198.
expected_nll: 0.225
