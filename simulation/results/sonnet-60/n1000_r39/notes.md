# Notes
Data: y vs x0 shows a lower branch (smooth nonlinear mean, sd ~0.3 -> 0.1) and an upper branch
(roughly linear in x, appearing for x0 > ~0.6, bimodal at high x).
Model: 2-component Normal mixture. Lower mean = quartic in x, upper mean = linear, log-sd of each
= quadratic in x, mixing logit = cubic in x. 17 fitted parameters, fit by ML (BFGS, multi-start).
Validation: 5-fold CV (shuffled) of the whole fitting procedure. Mean held-out NLL:
cubic mean/const sd 0.320; quartic + linear log-sd 0.257; + logit quadratic 0.241; logit cubic 0.237;
final config (quartic, linear, logit cubic, log-sd quadratic) 0.233. Gains beyond that were within noise,
so I stopped at this size. In-sample NLL of the final fit is 0.211.
Expected NLL is the CV figure plus a small allowance for selection among configurations.

expected_nll: 0.24
