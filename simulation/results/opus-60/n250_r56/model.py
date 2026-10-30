from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = -2.051235 + 4.579229 * t
    m2 = 0.201742 + t * (-12.092925 + t * (42.143909 + t * (-45.379881 + t * 14.159416)))
    s2 = exp(-0.902055 + -1.176623 * t)
    w = sigmoid(-3.570232 + 4.05034 * t)
    return Mixture([w, 1 - w], [Normal(m1, exp(-1.950708)), Normal(m2, s2)])
