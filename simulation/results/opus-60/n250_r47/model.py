from adr.dist import Normal, Mixture, exp, log, sigmoid


def model(x):
    """Cubic trunk that bifurcates (softplus kink near x0=0.613) into two branches;
    log-sd moves from a trunk level to per-branch levels through a sigmoid gate."""
    t = x[0]
    f = -0.0448 + t * (-7.1847 + t * (19.2023 + t * (-8.6682)))
    h = 0.07489 * log(1.0 + exp((t - 0.613) / 0.07489))
    g = sigmoid((t - 0.613) / 0.27274)
    up = Normal(f - 1.9525 * h, exp((1.0 - g) * -0.7594 + g * -2.7141))
    lo = Normal(f - 10.996 * h, exp((1.0 - g) * -0.7594 + g * -2.3344))
    return Mixture([0.52461, 0.47539], [up, lo])
