from adr.dist import Normal, Mixture, exp, maximum


def model(x):
    """Trunk that splits at x = c into two equally weighted branches; shared log-quadratic sd."""
    x = x[0]
    m = -0.010962 + x * (-8.591369 + x * (30.571714 + x * (-32.058632 + x * 9.101885)))
    g = maximum(x - 0.198374, 0.0)
    m2 = m + g * (-2.187297 + 8.220079 * g)
    s = exp(-0.974260 + x * (-0.379353 - 1.031453 * x))
    return Mixture([1.0, 1.0], [Normal(m, s), Normal(m2, s)])
