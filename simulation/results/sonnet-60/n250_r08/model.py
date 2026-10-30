from adr.dist import Normal, Mixture, sigmoid, exp


def model(x):
    t = x[0]
    m1 = ((((16.228 * t + -47.0143) * t + 40.2471) * t + -10.4424) * t + 0.0233)
    m2 = (-1.4767 * t + 7.1211) * t + -3.0783
    s1 = exp(-0.9044 + -1.324 * t)
    s2 = exp(-1.9897 + 0.1889 * t)
    p = sigmoid(-3.7318 + (7.8328 + -3.9239 * t) * t)
    return Mixture([1 - p, p], [Normal(m1, s1), Normal(m2, s2)])
