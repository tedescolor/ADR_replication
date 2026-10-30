from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = 0.4509 + t * (-15.7354 + t * (48.5878 + t * (-45.7118 + t * 14.9725)))
    m2 = -0.333 + t * (-3.9513 + t * (15.8019 + t * (-14.8557 + t * 2.3655)))
    s1 = exp(-1.4461 - 0.8005 * t)
    s2 = exp(-1.0625 - 1.0807 * t)
    w = sigmoid(-0.1425)
    return Mixture([w, 1 - w], [Normal(m1, s1), Normal(m2, s2)])
