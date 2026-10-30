from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    """Two-branch mixture: a curved lower branch everywhere, plus a linear upper branch
    switched on smoothly around x0 = 0.646."""
    t = x[0]
    m1 = -0.0216 + t * (-8.2622 + t * (28.9494 + t * (-29.4523 + t * 7.7853)))
    s1 = exp(-0.855 - 1.4069 * t)
    m2 = -2.3027 + 4.9445 * t
    s2 = exp(-2.2243)
    w = sigmoid(0.114) * sigmoid(50 * (t - 0.6461))
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
