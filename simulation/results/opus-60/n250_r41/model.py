from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = -0.0651 + t * (-6.9198 + t * (18.8924 - 9.3953 * t))
    s1 = exp(-0.8745 - 1.2826 * t)
    m2 = -6.0607 + t * (17.6762 - 12.5794 * t)
    s2 = exp(-2.1469)
    w2 = sigmoid(-22.2512 + t * (58.8201 - 38.5493 * t))
    return Mixture([1 - w2, w2], [Normal(m1, s1), Normal(m2, s2)])
