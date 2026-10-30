from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = 0.1167 + t * (-7.7871 + t * (20.0021 + t * -9.7931))
    s1 = exp(-1.0033 + -1.2017 * t)
    m2 = -5.1124 + t * (15.5257 + t * -11.372)
    s2 = exp(-0.985 + -1.4386 * t)
    w = sigmoid(-4.0414 + 5.5865 * t)
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
