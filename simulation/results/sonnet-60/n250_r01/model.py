from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    ml = 0.04614 + t * (-9.129 + t * (32.64 + t * (-34.69 + t * 10.17)))
    mu = -2.108 + 4.705 * t
    s1 = exp(-0.9332 + -1.062 * t)
    s2 = exp(-0.4713 + -1.593 * t)
    p = sigmoid(-3.943 + 4.556 * t)
    return Mixture([p, 1 - p], [Normal(mu, s2), Normal(ml, s1)])
