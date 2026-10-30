from adr.dist import Normal, Mixture, exp, maximum, sigmoid


def model(x):
    t = 2 * x[0] - 1
    m1 = -0.0978 + t * (1.3839 + t * (-1.3477 + t * (-2.1648 + t * (1.7917 + t * (0.2360 - 0.8233 * t)))))
    s = exp(-1.3930 + t * (-0.7198 - 0.2672 * t))
    u = maximum(x[0] - 0.4498, 0)
    m2 = m1 + u * (1.6023 + 8.8564 * u)
    w = sigmoid(-0.2188)
    return Mixture([w, 1 - w], [Normal(m1, s), Normal(m2, s)])
