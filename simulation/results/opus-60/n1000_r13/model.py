from adr.dist import Normal, Mixture, exp, sigmoid, maximum


def model(x):
    """Two-branch Gaussian mixture: cubic-spline means, shared log-quadratic sd."""
    u = x[0]
    t = 2 * u - 1
    h1 = maximum(u - 0.25, 0) ** 3
    h2 = maximum(u - 0.5, 0) ** 3
    h3 = maximum(u - 0.75, 0) ** 3
    m1 = 0.441818 - 7.63663 * u + 11.8164 * u**2 + 6.05148 * u**3 - 20.1205 * h1 + 1.4758 * h2 + 10.9341 * h3
    m2 = -0.409998 - 5.13717 * u + 11.7722 * u**2 + 4.42748 * u**3 - 37.7376 * h1 + 36.9514 * h2 - 22.319 * h3
    s = exp(-1.47304 - 0.517477 * t - 0.454987 * t**2)
    w = sigmoid(-0.260504 + 0.517389 * t)
    return Mixture([w, 1 - w], [Normal(m1, s), Normal(m2, s)])
