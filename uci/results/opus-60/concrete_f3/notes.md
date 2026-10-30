# Notes

## Data
824 rows, 8 nonnegative covariates (x0..x7) with small jitter; y is positive (2–82) and
spread a lot. x7 is heavily right-skewed with clustered values, so log(x7) is used. The
three variables x0, x1, x2 act like the components of a total, so ratios to b = x0+x1+x2
are used as features.

## Validation
10-fold CV (repeated with different shuffle seeds). Every fold standardises the inputs and
y on its own training split, fits by penalised maximum likelihood (L-BFGS, analytic
gradients, 3 random restarts, best training objective kept) and scores the mean NLL on the
held-out fold.

| model | CV NLL |
|---|---|
| linear mean + log-linear scale, Normal (13 engineered features) | 3.31 |
| same, Student-t | 3.30 |
| tanh MLP on raw inputs (log x7), H=4–6, Normal | 3.11–3.24 |
| MLP, ratio inputs C (6 inputs), H=6, t | 3.08 |
| MLP, ratio inputs B, H=4, t | 3.10 |
| **MLP, ratio inputs B, H=6, Student-t, L2 1e-2** | **2.996 (seed 0), 3.006 (seeds 1,2)** |
| same, L2 3e-3 / 5e-3 / 3e-2 | 3.066 / 3.045 / 3.081 |
| alternative input sets D, E (H=6, t) | 3.051 / 3.048 |

The Student-t likelihood was consistently better than the Normal (fitted df ≈ 3.8, so the
tails are heavy). H=7 would hit the 80-constant limit and go past 500 AST nodes.

## Final program
Inputs z = (log x7, x3/b, x1/b, x2/b, x4/b, x0, x5, x6), with b = x0+x1+x2.
There are 6 tanh hidden units. Both the location and the log-scale are linear in the
hidden units, and the output is StudentT(df, mu, exp(ls)). Input/output standardisation is
folded into the weights. That gives 69 constants and 461 AST nodes. It was refit on all
data with 10 restarts, which reached the same optimum as 3 restarts (train NLL 2.850). The
tanh units are bounded, so the scale stays finite and positive everywhere in the domain
box (b ≥ 99 there), and the contract check passes.

The CV estimate is about 3.00. The full-data fit uses about 11% more data, which helps a
little; picking the best of several configurations biases the estimate down a little.
Net estimate:

expected_nll: 3.01
