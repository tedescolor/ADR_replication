# Notes

Data: 85 rows; x0..x3 sit near design levels (x0 ~10-30, x1 ~12-25, x2 ~100/200/300, x3 20-100),
x4 is continuous (0.4-2.2, domain up to 2.57). y > 0 (0.05-0.61), right-skewed; corr(y,x4)=0.73, corr(y,x2)=-0.70.

Validation: 5-fold CV repeated 10-20 times (different shuffles), mean held-out NLL on the y scale.

Tried (CV NLL):
- LogNormal, linear in log-covariates: all -1.243, {1,2,4} -1.250, {0,1,2,4} -1.261; adding (log x3)^2 ~ -1.26
- LogNormal with heteroscedastic log-sigma: worse (-1.07 to -1.21)
- Log-Student-t: worse (~-1.23 to -1.25)
- Normal with sd proportional to mean: -1.22 to -1.26
- Gamma, log link: {1,2,4} -1.273, {0,1,2,4} -1.2765, all -1.260, +x3 quadratic -1.279,
  +raw x4 -1.290 (but bends the mean down for x4 > 2, i.e. in the extrapolation region), raw covariates -1.24

Choice: Gamma GLM, log link, log x0, x1, x2, x4 (x3 adds nothing), constant shape (~11.1),
fitted by ML on all data. The variants that scored better differ by about 0.01, which is within CV noise,
and they extrapolate less safely to the domain edge (x4 up to 2.57), so I kept the simpler model.
Training NLL is -1.369; CV gives -1.277. Small optimism allowance for new data.

expected_nll: -1.26
