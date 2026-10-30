from adr.dist import Normal, Mixture, exp, maximum


def model(x):
    """Two-branch Gaussian mixture: a shared cubic curve g(x); past a split point the
    lower branch bends away as g(x) - a*(x - c)^p. Shared log-quadratic noise scale."""
    t = x[0]
    g = ((-10.3606 * t + 20.5517) * t - 7.77) * t + 0.0505
    s = exp((-1.1185 * t - 0.4511) * t - 0.9482)
    w = 0.45467
    d = 9.0582 * maximum(t - 0.5251, 0.0) ** 1.29058
    return Mixture([w, 1.0 - w], [Normal(g, s), Normal(g - d, s)])
