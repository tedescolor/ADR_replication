from adr.dist import Normal, StudentT, Mixture, sigmoid, exp


def model(x):
    t = x[0]
    # main branch: quartic mean, log-linear scale, Student-t tails
    m1 = -0.0171 + t * (-8.7175 + t * (29.9084 + t * (-29.9888 + t * 7.8738)))
    s1 = exp(-0.8291 - 1.7136 * t)
    # second (upper) branch: linear mean, log-linear scale
    m2 = -2.3632 + 4.9665 * t
    s2 = exp(-0.4561 - 1.849 * t)
    w = sigmoid(-7.6987 + t * (20.0279 - 12.5543 * t))
    return Mixture([1 - w, w], [StudentT(16.68, m1, s1), Normal(m2, s2)])
