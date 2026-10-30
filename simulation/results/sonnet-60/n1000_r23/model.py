from adr.dist import Normal, Mixture, sigmoid, sin, exp


def model(x):
    t = x[0]
    m1 = -0.263 + 0.539 * sin(7.732 * t + 2.781)
    m2 = -2.158 + 4.756 * t
    s1 = exp(-0.962 + -0.885 * t)
    s2 = exp(-0.983 + -1.182 * t)
    w = sigmoid(-3.5 + 4.159 * t)
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
