from adr.dist import Normal, Mixture, exp, sin, sigmoid


def model(x):
    t = x[0]
    m1 = 0.14422436 * sin(13.78306145 * t - 5.98679944) - 0.1943458 + t * (-6.6221793 + t * (21.3303776 - 15.7625993 * t))
    s1 = exp(-0.83043945 - 1.44317621 * t)
    m2 = -2.00241888 + 4.56100367 * t
    s2 = exp(-2.1048058)
    w = sigmoid(-20.23606538 + t * (48.83210101 - 28.64650814 * t))
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
