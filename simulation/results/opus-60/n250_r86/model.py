from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = -0.11543 + t * (-5.9504 + t * (18.462 - 13.473 * t))
    s1 = exp(-1.0809 + t * (0.72593 - 2.2411 * t))
    m2 = -2.4735 + 5.1088 * t
    s2 = exp(-1.9263)
    w2 = 0.58743 * sigmoid(7.8799 * (t - 0.51592))
    return Mixture([1 - w2, w2], [Normal(m1, s1), Normal(m2, s2)])
