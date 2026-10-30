from adr.dist import Normal, Mixture, exp, maximum


def model(x):
    """Shared cubic trend that splits at x0 = 0.5 into an upper and a lower branch
    (equal-weight two-Gaussian mixture); log-sd linear in x0, shrinking along each branch."""
    t = 2.0 * x[0] - 1.0
    h = maximum(x[0] - 0.5, 0.0)
    base = ((-1.4050886 * t + 1.08736448) * t + 2.48355942) * t + 0.02832687
    ls = -0.3265013 * t - 1.35502647
    up = Normal(base + (-0.2983487 + 1.79611189 * h) * h, exp(ls - 0.93242378 * h))
    lo = Normal(base + (-3.78346454 - 5.19183619 * h) * h, exp(ls - 1.32861252 * h))
    return Mixture([0.5, 0.5], [up, lo])
