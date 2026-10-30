"""(y0, y1) = p(y0) p(y1 | y0).
y0 ~ LogNormal: log-location linear in log x0, log(1+x1), log(1+x2), x3, x4, x4^2, log x5;
log-scale linear in log x0, x3, x4.
y1 | y0 ~ mixture of a StudentT (location linear in the same covariates plus y0 and log y0,
log-scale linear in y0) and a narrow Normal near zero (about 5% of rows)."""
from adr.dist import LogNormal, StudentT, Normal, Mixture, Sequential, log, exp, minimum, maximum


def model(x):
    mu = 1.062 + 0.06389 * log(x[0]) - 0.007055 * log(1 + x[1]) + 0.05638 * log(1 + x[2]) + 0.01974 * x[3] - 0.03562 * x[4] + 0.1732 * log(x[5]) + 0.00185 * x[4] * x[4]
    sig = exp(minimum(maximum(-0.8377 - 0.1027 * log(x[0]) - 0.1422 * x[3] - 0.01953 * x[4], -4.0), 1.0))
    base = 1.014 + 0.09718 * log(x[0]) + 0.05516 * log(1 + x[1]) + 0.02557 * log(1 + x[2]) + 0.03621 * x[3] - 0.06545 * x[4] + 0.179 * log(x[5]) + 0.002166 * x[4] * x[4]

    def cond(y0):
        main = StudentT(9.386, base + 0.4073 * y0 - 1.158 * log(y0),
                        exp(minimum(-0.9004 + 0.05537 * y0, 2.0)))
        return Mixture([0.9507, 0.04925], [main, Normal(0.2003, 0.09632)])

    return Sequential(LogNormal(mu, sig), cond)
