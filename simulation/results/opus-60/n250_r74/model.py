from adr.dist import Normal, Mixture, exp, log


def model(x):
    """Two equal-weight branches: a cubic curve, and the same curve plus a softplus
    ramp that switches on near x=0.66. Shared log-linear noise scale."""
    u = x[0]
    g = ((-8.9055 * u + 17.4696) * u - 5.9285) * u - 0.2065
    k = exp(2.1797)
    h = log(1.0 + exp((u - 0.6642) * k)) / k
    s = exp(-0.894 - 1.4534 * u)
    return Mixture([0.5, 0.5], [Normal(g, s), Normal(g - 9.9333 * h, s)])
