from adr.dist import Mixture, Normal, exp, sigmoid


def model(x):
    t = x[0]
    m1 = 0.0754 + t * (-7.6341 + t * (19.4961 + t * -9.4254))
    m2 = -4.4832 + t * (13.8156 + t * -10.2795)
    s1 = exp(-0.979 + -1.1915 * t)
    s2 = exp(-0.3127 + -2.0162 * t)
    q = sigmoid(-3.9497 + 4.7687 * t)
    return Mixture([1 - q, q], [Normal(m1, s1), Normal(m2, s2)])
