from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = -0.159 + t * (-7.1862 + t * (19.8626 - 10.0185 * t))
    m2 = -0.2497 + t * (-4.1752 + t * (14.3173 - 10.8986 * t))
    s1 = exp(-0.9569 - 1.2413 * t)
    s2 = exp(-0.8612 - 1.2934 * t)
    w2 = sigmoid(-1.138 + 1.305 * t)
    return Mixture([1.0 - w2, w2], [Normal(m1, s1), Normal(m2, s2)])
