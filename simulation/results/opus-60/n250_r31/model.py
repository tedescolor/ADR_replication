from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = 2 * x[0] - 1
    mu1 = -0.032 + 2.585 * t + 1.365 * t * t - 1.455 * t * t * t
    mu2 = 0.014 + 1.187 * t - 2.102 * t * t
    sd1 = exp(-1.474 - 0.645 * t)
    sd2 = exp(-1.202 - 1.123 * t)
    w2 = sigmoid(-1.527 + 2.22 * t)
    return Mixture([1 - w2, w2], [Normal(mu1, sd1), Normal(mu2, sd2)])
