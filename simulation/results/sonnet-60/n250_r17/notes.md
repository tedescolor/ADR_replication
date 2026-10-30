# Notes
Data: y vs x0 in [0,1]. Single branch for x<0.65 (mean dips then rises), then forks into an upper branch (rising to ~2.5) and lower branch (falling to ~-0.9), with growing spread. 
Model: 2-component Normal mixture; comp1 mean cubic in x, log-sd linear in x; comp2 mean quadratic, log-sd linear; weight of comp2 = sigmoid(linear in x). 12 constants fitted by ML (L-BFGS, multi-start).
Validation: 5-fold CV (random folds), mean NLL on held-out folds, refit per fold.
CV NLL: (cubic,quad)=0.243 [chosen]; (cubic,linear)=0.324; (quartic,quad)=0.247 (quartic,cubic)=0.247; (quartic,quartic)=0.260; (quad,quad)=0.395. Train NLL of final 0.171.
Choice is lightly selected among 6 candidates, so expect slight optimism.
expected_nll: 0.26
