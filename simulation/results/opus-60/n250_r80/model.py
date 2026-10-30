from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    z = x[0] - 0.5
    m1 = 0.0551323 + z*(5.24246 + z*(5.00017 + z*(-11.6984 + z*(0.553876))))
    m2 = -0.088862 + z*(2.62327 + z*(-4.59648 + z*(-12.3329 + z*(8.19632))))
    s1 = exp(-1.43449 + z*(-1.2076 + z*(-2.54181)))
    s2 = exp(-1.32761 + z*(-0.396915 + z*(-3.68827)))
    w = sigmoid(0.381688 + -0.497646 * z)
    return Mixture([w, 1 - w], [Normal(m1, s1), Normal(m2, s2)])
