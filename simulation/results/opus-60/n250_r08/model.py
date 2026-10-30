from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = -0.11 + t * (-6.51 + t * (18.439 - 9.329 * t))
    s1 = exp(-0.968 - 1.054 * t)
    m2 = -3.839 + t * (12.378 - 9.494 * t)
    s2 = exp(-1.89)
    w = sigmoid(16.451 * (t - 0.296)) * sigmoid(-0.016)
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
