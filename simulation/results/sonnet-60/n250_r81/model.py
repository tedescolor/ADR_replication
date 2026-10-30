from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = -0.21379 + t * (-3.3938 + t * (5.162 + t * (9.4699 + t * -8.6465)))
    m2 = -3.8477 + t * (12.15 + t * -9.2654)
    s1 = exp(-0.89985 + -1.3546 * t)
    s2 = exp(-0.42513 + -1.6943 * t)
    w = sigmoid(-3.4674 + 4.178 * t)
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
