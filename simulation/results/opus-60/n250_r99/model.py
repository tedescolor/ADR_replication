from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = ((((-1.2609 * t - 1.0193) * t - 3.2108) * t + 13.768) * t - 5.6364) * t - 0.1264
    s1 = exp((-2.0405 * t + 0.8168) * t - 1.2976)
    m2 = (-8.341 * t + 10.5588) * t - 3.1351
    s2 = exp(-2.3328 * t - 0.1086)
    w = sigmoid(0.3389) * sigmoid(11.7155 * (t - 0.5781))
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
