from adr.dist import Normal, Mixture, exp, sigmoid, maximum


def model(x):
    """Fork: one curve that splits at t = c; upper branch continues the base curve,
    lower branch peels off by a power of (t - c). Shared log-quadratic noise sd."""
    t = x[0]
    up = (((-10.971846172838434) * t + 21.56461658514166) * t + -8.206144474167152) * t + 0.09635527564009834
    r = maximum(t - 0.4611144204789744, 0.0)
    lo = up - (8.930828957688007 * r ** 1.518972198962853)
    s = exp(((-0.9101310499547243) * t + -0.5473499772474267) * t + -0.9523774223119776)
    w = sigmoid(0.032775529379887036)
    return Mixture([w, 1 - w], [Normal(up, s), Normal(lo, s)])
