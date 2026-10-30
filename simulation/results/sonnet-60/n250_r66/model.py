from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = -0.04275 + t * (-5.825 + t * (17.23 + t * -12.36))
    m2 = -0.136 + t * (-6.571 + t * (18.28 + t * -9.045))
    s1 = exp(-0.7279 + -1.673 * t)
    s2 = exp(-0.9306 + -1.306 * t)
    u = sigmoid(1.732 + -1.904 * t)
    return Mixture([1 - u, u], [Normal(m1, s1), Normal(m2, s2)])
