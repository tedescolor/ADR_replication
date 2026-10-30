from adr.dist import Normal, Mixture, sigmoid, exp


def model(x):
    t = x[0]
    m1 = -0.7998 + t * (-0.8483 + t * (8.7624 - 8.0994 * t))
    m2 = -0.0196 + t * (-7.7781 + t * (20.8721 - 10.5921 * t))
    s1 = exp(-0.4171 - 2.0124 * t)
    s2 = exp(-0.8468 - 1.3842 * t)
    w = sigmoid(-4.3246 + t * (11.1897 - 7.1879 * t))
    return Mixture([w, 1 - w], [Normal(m1, s1), Normal(m2, s2)])
