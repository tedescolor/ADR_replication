from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = -0.2611 + t * (-2.9045 + t * (-7.3465 + t * (57.493 + t * (-71.068 + t * 26.641))))
    s1 = exp(-1.0429 - 1.2533 * t)
    m2 = -3.4382 + t * (11.424 - 8.9044 * t)
    s2 = exp(0.15949 - 2.4398 * t)
    w = sigmoid(-9.2128 + t * (24.122 - 15.581 * t))
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
