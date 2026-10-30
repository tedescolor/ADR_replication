from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = 2 * x[0] - 1
    m1 = -0.032028 + 2.543*t + 1.2858*t**2 - 1.3143*t**3 + 0.054657*t**4
    m2 = -0.1014 + 0.98216*t - 0.66617*t**2 - 1.162*t**3
    s1 = exp(-1.4568 - 0.53178*t)
    s2 = exp(-1.4646 - 0.80874*t)
    w = sigmoid(0.26272 - 1.6899*t + 2.1761*t**2)
    return Mixture([w, 1 - w], [Normal(m1, s1), Normal(m2, s2)])
