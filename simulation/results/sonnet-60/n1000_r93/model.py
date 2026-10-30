from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = 0.03111 + t * (-7.743 + t * (20.6 + t * (-10.71 + t * 0.312)))
    m2 = -3.391 + t * (11.06 + t * -8.597)
    s1 = exp(-0.8488 + -1.363 * t)
    s2 = exp(-0.4429 + -1.879 * t)
    p = sigmoid(6.662 + t * (-17.0 + t * 10.64))
    return Mixture([p, 1 - p], [Normal(m1, s1), Normal(m2, s2)])
