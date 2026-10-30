# Notes
Data: 1 covariate on [0,1]; scatter shows a bimodal conditional at high x (two branches: one rising to ~2.5, one falling to ~-1) merging near x~0.5, with x-dependent noise.
Model: 2-component Normal mixture. Component means are polynomials in x (quartic and quadratic), log-sd linear in x, mixing logit quadratic in x. 15 constants.
Validation: 4-fold CV (shuffled) of full ML fits. Candidates: mean degrees (3..5, 1..3), sd degrees 1-2, weight degrees 1-3.
Mean CV NLL: ~0.200-0.205 for most configs from (4,2,1,2) upward; cubic first component or linear weight were clearly worse (~0.25-0.31 on a single split). I picked the small (4,2,1,2) config (CV 0.2021), since larger ones gave no real gain.
Final fit on all data: train NLL 0.189; CV NLL 0.202.
expected_nll: 0.205
