from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    u = x[0]
    m1 = ((-9.6849 * u + 19.652) * u + -7.4709) * u + 0.017053
    m2 = (-9.3032 * u + 12.122) * u + -3.7515
    s1 = exp((-0.46737 * u + -0.85468) * u + -0.95594)
    s2 = exp(-1.446 * u + -0.79175)
    p = sigmoid(4.5336 * u + -3.8242)
    return Mixture([1 - p, p], [Normal(m1, s1), Normal(m2, s2)])
