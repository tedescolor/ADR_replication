# Notes
Data: x0 in [0,1], y heteroscedastic, nonlinear mean; for x0 > ~0.6 y splits into two branches (one rising to ~2.5, one falling to ~-0.9).
Validation: 10-fold CV (2 repeats, random splits, refit incl. multi-start) of mean NLL.
- single Normal, poly mean + poly log-sd: best 0.498 (cubic mean, quadratic log-sd); StudentT / SkewNormal no gain.
- 2-component Normal mixture, logistic weight in x, polynomial means / log-sds (degrees mean A,logsd A,mean B,logsd B,weight):
  (3,1,1,0,1) 0.260; (3,1,2,0,1) 0.160; (3,1,2,0,2) 0.136; (3,1,2,1,2) 0.130; (3,1,3,1,2) 0.129; (4,1,2,1,2) 0.144.
Final: (3,1,2,1,2), 14 parameters, fit by ML on all data (train NLL 0.066). Extra complexity gave no CV gain.
Estimate allows for slight selection optimism and train/CV gap.
expected_nll: 0.14
