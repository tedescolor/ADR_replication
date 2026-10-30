from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = (((10.215 * t - 35.8684) * t + 34.7041) * t - 10.2232) * t + 0.1638
    m2 = ((-9.6086 * t + 19.017) * t - 6.7541) * t - 0.1546
    s1 = exp(-1.3756 * t - 0.7728)
    s2 = exp(-1.088 * t - 1.057)
    w = sigmoid(0.4536 - 0.6061 * t)
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
