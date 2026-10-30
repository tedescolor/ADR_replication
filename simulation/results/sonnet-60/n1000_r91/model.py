from adr.dist import Mixture, Normal, exp, sigmoid


def model(x):
    t = x[0]
    m1 = ((((16.10544) * t + -49.35509) * t + 44.37571) * t + -11.95821) * t + -0.12831
    m2 = (((-10.59015) * t + 21.36495) * t + -8.54707) * t + 0.25589
    s1 = exp(((-1.01142) * t + 0.08742) * t + -1.3795)
    s2 = exp(((-1.2187) * t + 0.00102) * t + -1.16278)
    wu = sigmoid(1.35417 + -1.89249 * t)
    return Mixture([1 - wu, wu], [Normal(m1, s1), Normal(m2, s2)])
