from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    """Two-branch Gaussian mixture: cubic lower branch, linear upper branch, x-dependent weight."""
    t = x[0]
    p = sigmoid(((-12.43762) * t + 19.22407) * t + -7.31719)
    lower = Normal((((-14.81336) * t + 20.94684) * t + -7.23712) * t + 0.02213, exp((-1.35421) * t + -0.90213))
    upper = Normal((5.01555) * t + -2.36258, exp((-1.21155) * t + -1.01852))
    return Mixture([1.0 - p, p], [lower, upper])
