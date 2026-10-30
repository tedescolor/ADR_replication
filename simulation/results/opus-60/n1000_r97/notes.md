# Notes

## Data
There are 1000 rows with x0 in [0,1]. For x < ~0.45, y follows one curve (it dips to about -0.8 near x=0.25) with noise sd falling from ~0.38 to ~0.2. Above that point the data split into two branches like a pitchfork. The lower branch rises to ~0.15 and then falls to -1 at x=1. The upper branch rises almost linearly to 2.5. Each branch has about half the points (~46% upper), and both have noise sd falling toward ~0.1 at x=1.

## Validation
5-fold CV, with the same random fold split (seed 1) for every candidate, and the mean held-out NLL pooled over all rows. Fits are maximum likelihood (L-BFGS-B then BFGS) with a few random restarts. One restart in each fold is warm-started from the full-data fit, which leaks a little; the impact should be small, because this only picks the basin of the optimum.

## Candidates (CV NLL)
- Free 2-component Gaussian mixture: polynomial means, log-sd as polynomials, weight = p*sigmoid(k(x-c)). Best about 0.212–0.218.
- Pitchfork: shared lower mean m(x) (polynomial of degree 6), upper mean m(x)+d(x) with d = a v + b v^2, v = max(x-c,0), and a constant weight (the components coincide before c, so the weight does not matter there):
  - shared log-linear sd: 0.2001 (13 params). Degree 5: 0.2034. Degree 7: 0.2003. Degree 8: 0.2034. Cubic d: 0.2022. Softplus hinge: 0.2008. Quadratic log-sd: 0.2000.
  - **separate log-linear sd per branch: 0.1983 (15 params)** ← chosen
  - Student-t components: 0.2005 (shared sd) and 0.1991 (separate sd), so no gain.
- Differences of about 0.002 are within noise. I chose the separate-sd pitchfork because it scored best with both Normal and Student-t components, and it is still small.

## Final program
A 2-component Normal mixture with a constant weight (sigmoid(-0.154) ≈ 0.46). The lower mean is a degree-6 polynomial. The upper mean is that polynomial plus a quadratic in the hinge max(x-0.244,0). Each component has sd = exp(linear in x). That is 15 fitted values, written as 16 literals (counting the hinge's 0.0). Training NLL is 0.1827 and 5-fold CV NLL is 0.1983. I added a small allowance for the warm-start leakage and for selecting among models.

expected_nll: 0.203
