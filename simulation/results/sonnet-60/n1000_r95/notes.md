# Notes
Data: 1 covariate x0 in [0,1]. Scatter plot shows a smooth lower branch (sinusoidal in x0, noise sd shrinking with x0) and, for x0 > ~0.6, a second upper branch (quadratic mean), so y|x is bimodal at high x.
Model: 2-component Normal mixture. Branch A: mean a + b*sin(w*x+phi), sd = exp(linear). Branch B: quadratic mean, sd = exp(linear). Weight of B = sigmoid(k*(x-x0)). 15 fitted constants (+ a few structural 1s), well under the limit.
Fitting: maximum likelihood (Nelder-Mead then BFGS). Validation: 5-fold CV with random folds, refitting all parameters each fold: fold NLLs 0.306, 0.273, 0.332, 0.210, 0.288, mean 0.282. In-sample NLL 0.269.
Not tried: richer weight function (e.g. quadratic logit), more components; the CV gap to in-sample is small, so I kept the simple form.
expected_nll: 0.29
