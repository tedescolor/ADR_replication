from adr.dist import Normal, Mixture, sigmoid, exp


def model(x):
    t = x[0]
    m1 = 0.1509 + t * (-10.3652 + t * (35.7022 + t * (-37.7227 + t * 11.2611)))
    s1 = exp(-0.8061 - 1.4286 * t)
    m2 = -2.4119 + 5.0169 * t
    s2 = exp(-0.7684 - 1.4106 * t)
    w2 = 0.5 * sigmoid(73.55 * (t - 0.2129))
    return Mixture([1.0 - w2, w2], [Normal(m1, s1), Normal(m2, s2)])
