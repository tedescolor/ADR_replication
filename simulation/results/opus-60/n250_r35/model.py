"""Two-branch mixture: a smooth cubic trend that splits into two Gaussian branches past a soft hinge."""
from adr.dist import Normal, Mixture, exp, log, abs, maximum


def model(x):
    t = x[0]
    f = -0.0555 + t * (-7.36377 + t * (19.87929 + t * -10.22517))
    z = (t - 0.63664) / 0.08264
    h = 0.08264 * (maximum(z, 0.0) + log(1.0 + exp(-abs(z))))
    sd = exp(-0.80662 + -1.32249 * t)
    return Mixture([0.54738, 0.45262], [Normal(f + 0.72476 * h, sd), Normal(f - 8.8624 * h, sd)])
