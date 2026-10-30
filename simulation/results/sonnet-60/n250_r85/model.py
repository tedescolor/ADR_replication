from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = (((16.248 * t - 48.765) * t + 43.391) * t - 12.026) * t + 0.196
    s1 = exp(-0.864 - 1.201 * t)
    m2 = -2.368 + 5.022 * t
    p = sigmoid(-3.705 + 3.942 * t)
    return Mixture([1 - p, p], [Normal(m1, s1), Normal(m2, exp(-1.963))])
