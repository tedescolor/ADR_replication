# Notes

**Data.** 250 rows, x0 in (0,1). For x < ~0.5 there is one noisy, wavy curve (sd ~0.3).
Above ~0.5 the data splits into two branches. The upper one is close to linear
(y ≈ 4.6x − 2.04, reaching 2.6 at x=1) and very tight. The lower one continues the
wavy curve, peaks near x≈0.7 and falls to −1 at x=1.

**Model family.** A 2-component Normal mixture where every parameter depends on x:
the means are polynomials, the sds are exp(linear), and the gate is a logistic in x.
I fitted by maximum likelihood (BFGS with restarts).

**Validation.** 5-fold CV repeated twice (fixed seeds), refitting each fold. The final model
also got 10-fold × 3 CV.

| variant | train NLL | CV NLL |
|---|---|---|
| comp A = upper+main (quartic), B = lower branch (quadratic) | 0.166 | 0.222 |
| same, A quintic/B cubic | 0.113 | 0.185 |
| **A = main curve quintic, B = linear upper branch, logistic gate** | 0.129 | **0.183** (10x3: 0.182) |
| same, A degree 6 | 0.123 | 0.182 |
| same, quadratic log-sd for A | 0.128 | 0.187 |
| same, constant sd for B | 0.147 | 0.204 |
| Student-t for A (df → ~4700, so it collapses to Normal) | 0.129 | 0.183 |
| gate with max weight < 1 (3-param gate) | 0.119 | 0.188 |
| A = linear + sinusoid | 0.132 | 0.184 |
| A = linear + sinusoid with linear amplitude, 3-param gate | 0.119 | 0.188 |

The differences between the top variants (~0.005) are within CV noise. I chose the
quintic main curve + linear upper branch + 2-parameter logistic gate. It is the simplest
model near the best CV score: 14 fitted constants (15 literals in total).

**Final.** Fit on all data: train NLL 0.1288. CV NLL is 0.182–0.183. The final fit
uses all 250 rows rather than 200–225, which should help a little, but some
model-selection optimism pulls the other way. Estimate:

expected_nll: 0.185
