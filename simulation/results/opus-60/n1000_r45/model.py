from adr.dist import Normal, Mixture, exp, maximum


def model(x):
    """Shared quartic trend that splits at x=0.567 into two branches (Gaussian mixture)."""
    t = x[0]
    m0 = (((-4.0853 * t - 5.4894) * t + 17.5613) * t - 6.8708) * t - 0.0355
    r = maximum(t - 0.5671, 0.0)
    up = m0 + (6.6684 * r + 0.3349) * r
    lo = m0 + (2.8558 * r - 5.9902) * r
    ls = -0.7697 * t - 0.9806
    return Mixture([0.4408, 0.5592],
                   [Normal(up, exp(ls - 1.0427 * r)), Normal(lo, exp(ls - 1.523 * r))])
