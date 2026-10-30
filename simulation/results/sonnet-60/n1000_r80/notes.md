# Notes
Data: x0 in (0,1), y bimodal at high x (a lower branch plus an upper branch rising to ~2.4); mean and spread vary nonlinearly in x.
Model: 2-component Gaussian mixture. Weight = sigmoid(quadratic in x). Component 1: quartic mean, log-sd linear. Component 2: quadratic mean, log-sd linear. 15+... = 17 constants (+ the 1 in 1-p).
Validation: 5-fold CV (random folds), refit with 5 random starts per fold. Mean NLL by (deg+1 of m1,m2,logsd1,logsd2,weight):
 (3,2,1,1,2) 0.473; (4,2,1,1,2) 0.348; (4,3,2,1,2) 0.271; (5,3,2,2,3) 0.235; larger variants 0.231-0.236 (plateau, so I kept the simpler one).
Final: (5,3,2,2,3) refit on all data (training NLL 0.213), passes contract check.
Since model selection used the same CV folds, the estimate is slightly optimistic.
expected_nll: 0.24
