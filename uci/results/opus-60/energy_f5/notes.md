# Notes

## Data
7656 rows, 4 covariates. y is strongly (negatively) related to x0 and x1, weaker to x2, x3.
OLS residual sd 4.55; residuals left-skewed and heavy-tailed (kurtosis 3.6).
kNN shows extra fine-grained structure: residuals vs x1 form many small clusters with
their own offsets (1-NN pair noise ~3.2 vs ~4.2 for polynomial means). That can't be captured
within the 80-constant / 500-AST-node budget, so the model uses smooth terms only.

## Validation
5-fold CV (fixed random split, seed 0). Each candidate is fit by full maximum likelihood
(L-BFGS, analytic gradients) on 4 folds and scored by mean NLL on the held-out fold.

| mean poly | log-scale poly | components | CV NLL |
|---|---|---|---|
| 2 | 0 | 1 (normal) | 2.872 |
| 2 | 2 | 2 | 2.823 |
| 3 | 0 | 1 | 2.851 |
| 3 | 0 | 2 | 2.815 |
| 3 | 1 | 2 | 2.807 |
| 3 | 2 | 2 | **2.8017** |
| 3 | 2 | 3 | 2.8019 |
| 3 | 3 | 2 | 2.8019 |
| 4 | 2 | 2 | 2.7876 (70 mean coefs, too many nodes) |

Making the outlier-component weight depend on x (linear or quadratic logit) gained at most
0.004, which wasn't worth the AST nodes.

## Term selection (final program)
Starting from cubic mean + quadratic log-scale, I ran stepwise selection on the full-data
likelihood: add up to 3 quartic mean monomials per round if each gains >6 in total
log-likelihood, then drop terms (mean or scale) costing <2.5. After 3 rounds: 35 mean
monomials (several quartic, e.g. z1 z3^3, z0 z3^3, z0^2 z1 z2; a few weak quadratic/cubic
terms removed) and 10 log-scale monomials.
5-fold CV NLL of selected structure: round 1 2.7904, round 3 **2.7866** (in-sample 2.7796).
Selection used all the data, so this CV is a little optimistic.

## Model
y | x ~ 0.991 N(m(x), s(x)) + 0.009 N(m(x) - 2.65 s(x), 3.58 s(x))
(a narrow main component plus a wide, left-shifted component that captures the skew and
heavy left tail). m is the selected polynomial in (x - centre) and log s the selected
quadratic. Both are written in nested Horner form to fit the 500-node limit (448 nodes, 53 constants).
Rounding the coefficients to 6 significant digits doesn't change the in-sample NLL (2.77961).
Passes the contract check, including the corners of the domain box.

## Estimate
CV 2.7866. I add ~0.005 for the selection optimism and the uncertainty in extrapolating
toward the domain edges.

expected_nll: 2.792
