from adr.dist import Normal, Mixture, sin, exp, sigmoid


def model(x):
    t = x[0]
    m1 = -0.279 + 0.579 * sin(7.732 * t + -3.512)
    s1 = exp(-1.013 + -0.932 * t)
    m2 = -1.469 + 2.956 * t + 1.146 * t * t
    s2 = exp(-1.103 + -1.075 * t)
    w2 = sigmoid(-2.672 + 2.85 * t)
    return Mixture([1 - w2, w2], [Normal(m1, s1), Normal(m2, s2)])
