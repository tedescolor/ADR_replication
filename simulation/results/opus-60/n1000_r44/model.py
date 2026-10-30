from adr.dist import Normal, Mixture, sigmoid


def model(x):
    t = x[0]
    c3 = -10.7132 * t * t * t
    m1 = 0.0162 + t * (-7.9747 + 21.1432 * t) + c3
    m2 = -0.3624 + t * (-3.7595 + 13.845 * t) + c3
    s = 0.3915 - 0.2963 * t
    w = sigmoid(1.8587 - 2.1704 * t)
    return Mixture([w, 1 - w], [Normal(m1, s), Normal(m2, s)])
