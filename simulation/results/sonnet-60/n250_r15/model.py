from adr.dist import Normal, Mixture, sigmoid, exp


def model(x):
    t = x[0]
    m1 = -0.096 + t * (-6.843 + t * (18.724 + t * -9.241))
    m2 = -3.908 + t * (12.397 + t * -9.443)
    s1 = exp(-0.891 + -1.139 * t)
    s2 = exp(-0.035 + -2.243 * t)
    p = sigmoid(-4.723 + 5.642 * t)
    return Mixture([1 - p, p], [Normal(m1, s1), Normal(m2, s2)])
