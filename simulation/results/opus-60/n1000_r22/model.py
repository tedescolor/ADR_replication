from adr.dist import Normal, Mixture, exp, maximum


def model(x):
    """Trunk curve that splits at x=0.513 into two equally likely branches."""
    t = x[0] - 0.5
    u = maximum(x[0] - 0.5129, 0.0)
    m = -0.0143 + 4.3446 * t + 1.4071 * t**2 - 14.708 * t**3
    g = -1.6615 * u + 3.6034 * u**2
    h = 2.0791 * u + 2.958 * u**2
    s = exp(-1.4924 - 1.4266 * t - 0.3843 * t**2)
    return Mixture([0.5, 0.5], [Normal(m + g + h, s), Normal(m + g - h, s)])
