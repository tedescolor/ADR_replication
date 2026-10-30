# Notes
Data: 1 covariate x0 in (0,1), y heteroscedastic with nonlinear mean; for x0 > ~0.55 y becomes clearly bimodal (two branches).
Validation: 5-fold CV (fixed random split), refit per fold, mean held-out NLL.
- Single Normal, poly mean/log-sd: best CV 0.617 (t, skew-normal gave no gain).
- Two-component Normal mixture, polynomial means, log-sd linear in t=2x-1, logistic weight quadratic in t: CV 0.246 (config 4,3,1,2). Neighbouring configs 0.246-0.25; more flexible ones no better.
Final: that mixture fit on all data (16 params + few structural constants), train NLL 0.232.
CV estimate 0.246 (slight selection optimism over ~16 configs), so I say 0.255.
expected_nll: 0.255
