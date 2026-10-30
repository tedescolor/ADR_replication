from adr.dist import Normal, Mixture, exp, maximum


def model(x):
    """Two-branch Gaussian mixture: branch B splits off branch A at t = 0.5103."""
    t = x[0]
    ma = (((-2.849 * t - 3.9648) * t + 15.7281) * t - 6.3988) * t - 0.0579
    mb = ma - maximum(t - 0.5103, 0.0) * (5.9647 * t + 1.1151)
    s = exp(-1.3613 * t - 0.8462)
    return Mixture([0.5217, 0.4783], [Normal(ma, s), Normal(mb, s)])
