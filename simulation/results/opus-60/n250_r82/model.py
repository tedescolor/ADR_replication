from adr.dist import Normal, Mixture, exp, sin, sigmoid


def model(x):
    t = x[0]
    m1 = 0.819 * sin(5.4224 * t - 2.436) + 0.283 - 1.34 * t
    m2 = 5.033 * t - 2.388
    s = exp(-1.05 - 1.078 * t)
    w = 0.5 * sigmoid(20.0 * (t - 0.33))
    return Mixture([1 - w, w], [Normal(m1, s), Normal(m2, s)])
