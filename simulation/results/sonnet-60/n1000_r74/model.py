from adr.dist import Mixture, Normal, sigmoid, exp


def model(x):
    t = x[0]
    m1 = -0.01906 + t * (-7.644 + t * (21.42 + t * (-12.61 + t * 1.366)))
    m2 = -5.1 + t * (15.34 + t * -11.2)
    l = -0.9038 + -1.283 * t
    p = sigmoid(-9.621 + t * (23.66 + t * -14.52))
    return Mixture([1 - p, p], [Normal(m1, exp(l)), Normal(m2, exp(l + -0.0575))])
