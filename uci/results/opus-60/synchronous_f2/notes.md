# Notes

## Data findings
- x1 + x2 ≈ 1 (residual sd 0.0015, carries no information about y), so x1 is dropped.
- For 391/445 rows, y = 1.1803 + 0.0009996·x3 almost exactly: residual sd 0.0029, roughly Gaussian (kurtosis -0.35).
  Adding x0 or x2 gives no meaningful gain.
- 54 rows (12%) have x3 in [-3.5, 4.9], a broken reading of about 0 (noise), while y is normal (1.26–1.92).
  The smallest valid x3 is 36.9, so the two groups are cleanly separated.
- For those rows y has to come from (x0, x2). Both lie on design grids: x0 in steps of 0.1, x2 in steps of 0.01.
  For x0 < 4.55, x2 takes even multiples of 0.01; for x0 > 4.55 it takes odd ones (two regimes).
  y has a sawtooth in x0 with period 0.5 and steps in x2 every 0.05, starting at 0.095 (flat below that).

## Model
Mixture gated on x3: w = 0.995·sigmoid(x3 − 15).
- Valid branch: Normal(1.1803 + 0.0009996·x3, 0.00294).
- Fallback (weight 1 − w, never below 0.005 as an outlier guard): Student-t (df ≈ 2.3).
  - The mean is linear in [x0, saw(2x0+0.4), step(x2), x2, 1{x0>4.55}] plus all 10 pairwise products.
  - log-scale is linear in x0.
  - saw(t) = t − round(t), written with arctan2(sin, cos).
  - The fallback is fitted by maximum likelihood on all 445 rows, since y | x0, x2 is the same whether or not x3 is broken.
- Size: 41 constants, 365 AST nodes.

## Validation
- 10-fold CV, repeated twice, of the fallback's Student-t fit.
- Variants compared:
  - with or without squared terms;
  - scale constant, linear in x0, with a regime term added, or also linear in x2.
- All variants were within about 0.02 nats, so I chose the simpler mean (no squares) with log-scale linear in x0.
- CV NLL of the fallback: −3.25 over all rows, −3.41 on the 54 broken-x3 rows.
  Residual sd is about 0.011, versus 0.048 for a plain cubic in (x0, x2).
- The valid-x3 branch has 3 parameters, and its in-sample NLL of −4.41 is essentially its CV value.
- Combined estimate: 0.879·(−4.405 + 0.005 gate cost) + 0.121·(about −3.33) ≈ −4.27.
  I round towards a more conservative value because of the small number of broken rows and possible test outliers.
- In-sample NLL of the final model is −4.296.

expected_nll: -4.25
