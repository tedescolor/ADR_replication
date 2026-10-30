from adr.dist import Mixture, Normal, exp, sigmoid


def model(x):
    t = x[0]
    m1 = -0.0439 + t * (-6.803 + t * (20.131 - 14.305 * t))
    s1 = exp(-0.848 - 1.2744 * t)
    m2 = -2.4668 + 5.1533 * t
    w = 0.4226 * sigmoid(80.0 * (t - 0.437))
    return Mixture([1.0 - w, w], [Normal(m1, s1), Normal(m2, 0.1705)])
