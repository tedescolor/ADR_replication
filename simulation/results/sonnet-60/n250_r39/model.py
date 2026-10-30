from adr.dist import Mixture, Normal, exp, sin, sigmoid


def model(x):
    t = x[0]
    m1 = -0.2797 + 0.5465 * sin(7.5931 * t + -3.4122)
    m2 = -2.6987 + 5.4311 * t
    s1 = exp(-0.9516 + -0.9693 * t)
    w = sigmoid(3.9212 * (t - 0.9457))
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, 0.1393)])
