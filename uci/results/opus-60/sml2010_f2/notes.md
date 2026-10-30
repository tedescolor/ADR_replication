# Conditional density model for y

## Exploration
- y is in [22, 84]. It is most correlated with x6 (0.82), x5, x19, x0 and x1. x16-x18 look like noise.
- Linear Gaussian model: 5-fold CV NLL 3.016. Adding pairwise products of the main covariates
  brings it to 2.78. Greedy forward selection over products picked up many small, scattered
  interactions, which suggests a smooth nonlinear surface rather than a few clear terms.
- Tried physically motivated ratios such as es(x2)/es(x0): they did not help.

## Validation
5-fold CV with a fixed permutation (seed 0). Every number below is the out-of-fold mean NLL on
the original y scale.

## Candidates (1-hidden-layer tanh MLP, Gaussian, fitted by L-BFGS with an L2 penalty on the weights)
- All 17 non-noise inputs, 10 units: 2.401 (about 200 constants, so too large for the 80-constant cap).
- Without x3/x4 the CV NLL got clearly worse (2.56), so x3/x4 matter.
- Within the budget:
  - [0,1,3,4,5,6,9,11,19] with 7 units: 2.515
  - same inputs with 6 units plus a linear skip: 2.504
  - [0,1,3,4,5,6,11,19] with 8 units (82 constants): 2.469
  - **[0,1,2,3,4,5,6,9,11,19] with 6 units, no skip (74 constants): 2.472** (chosen)
- Penalty: 1e-4 gave 2.472, 1e-5 gave 2.481, 3e-4 gave 2.570.
- Heteroscedasticity: |CV residual| has no correlation with any covariate above 0.05, so sigma is
  constant.
- Tails: the CV residuals have excess kurtosis of about 0.75. A Student-t with df=8, whose scale is
  fitted on the out-of-fold standardized residuals, gives about 2.459 (vs 2.472 for the Gaussian).

## Final program
6 tanh hidden units over x0..x6, x9, x11, x19, with the input standardization folded into the
weights. Mean = linear combination of the units. Noise: StudentT(df=8, scale=2.488).
75 literals, 355 AST nodes; passes the contract check. Training NLL is 2.404.
The final model was refit on all 2211 rows.

## Estimate
The CV estimate is about 2.46. Refitting on all the data should help a little, while the
selection among configurations and the optimizer's local optima (the fit is not deterministic)
add some optimism. Net estimate:

expected_nll: 2.47
