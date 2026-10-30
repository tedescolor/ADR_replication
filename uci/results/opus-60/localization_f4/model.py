from adr.dist import Gamma, log, exp


def model(x):
    """Gamma regression with log link on log-covariates (x3 dropped)."""
    eta = (5.43909 - 0.22332 * log(x[0]) - 0.72771 * log(x[1])
           - 0.82914 * log(x[2]) + 0.77035 * log(x[4]))
    shape = 11.10427
    return Gamma(shape, shape / exp(eta))
