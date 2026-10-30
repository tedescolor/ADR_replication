from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = 0.098467 + t * (-8.1847 + t * (21.49 + t * -10.919))
    m2 = -1.8058 + t * (2.6965 + t * (4.8308 + t * -6.7335))
    s1 = exp(-0.85942 + -1.407 * t)
    s2 = exp(-0.24587 + -2.1647 * t)
    w = sigmoid(6.4965 + t * (-16.906 + 10.793 * t))
    return Mixture([w, 1 - w], [Normal(m1, s1), Normal(m2, s2)])
