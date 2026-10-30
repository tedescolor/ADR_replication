from adr.dist import Mixture, Normal, exp, sigmoid


def model(x):
    t = x[0]
    m1 = -0.0714 + t * (-7.0303 + t * (21.816 - 15.9286 * t))
    m2 = -2.0813 + 4.6478 * t
    s1 = exp(-1.0587 - 1.0335 * t)
    s2 = exp(-0.168 - 2.35 * t)
    w2 = sigmoid(-4.0815 + 5.0852 * t)
    return Mixture([1 - w2, w2], [Normal(m1, s1), Normal(m2, s2)])
