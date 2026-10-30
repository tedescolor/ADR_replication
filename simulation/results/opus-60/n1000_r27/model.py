from adr.dist import Normal, Mixture, exp, log


def model(x):
    """Two-branch Gaussian mixture: a shared quartic curve m1, and an upper branch that
    departs from it via a softplus hinge near x=0.63; common heteroscedastic sd."""
    t = x[0] - 0.5
    h = 0.08958
    r = h * log(1.0 + exp((x[0] - 0.62509) / h))
    m1 = (((11.54132 * t - 14.88612) * t - 4.17105) * t + 2.70381) * t - 0.13612
    m2 = m1 + 9.14059 * r
    sd = exp((-0.44489 * t - 1.53565) * t - 1.48828)
    w = 0.47457
    return Mixture([w, 1.0 - w], [Normal(m1, sd), Normal(m2, sd)])
