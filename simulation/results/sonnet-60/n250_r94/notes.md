# Notes
Data: y vs x0 is bimodal at high x (two branches: one falling, one rising), heteroscedastic.
Model: 2-component Normal mixture; mean of comp A cubic in x, comp B quadratic; log-sd linear in x
for each; mixing weight logistic-linear in x. 13 fitted parameters (ML, 40 random restarts).
Validation: 5-fold CV (shuffled, seed 1), refit per fold with 4 restarts. Structures compared
(polynomial degrees of means/log-sd): (3,3,1) 0.61, (4,3,1) 0.37, (4,3,2) 0.294, (4,4,2) 0.299,
(5,3,2) 0.338, (4,2,2) 0.347. Chose (4,3,2). CV used fewer restarts than the final fit
(local optima), so the CV number is somewhat pessimistic; full-data train NLL is 0.236.
I use CV 0.29 plus a small optimism allowance for selection among structures.

expected_nll: 0.31
