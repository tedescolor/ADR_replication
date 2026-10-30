# Notes
Data: at x>0.7 y splits into two branches (upper rising, lower falling); below that one branch.
Model: 2-component Normal mixture; lower branch mean cubic in x, upper branch mean linear in x,
constant sigmas (0.273, 0.124), mixture weight sigmoid(linear in x). 12 constants.
Validation: 5-fold CV (random folds), multi-start MLE with tiny ridge. Mean CV NLL:
(cubic,linear) 0.265; (cubic,quad) 0.259; (quintic,quad) 0.250; (quad,linear) 0.550; quartic variants no better/unstable.
Chose cubic/linear for simplicity (gains from bigger models are within fold noise). Train NLL 0.218.
expected_nll: 0.27
