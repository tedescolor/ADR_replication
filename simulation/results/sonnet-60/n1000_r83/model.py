from adr.dist import Mixture, Normal, exp, sigmoid


def model(x):
    t = x[0]
    pw = sigmoid(-11.943 + t * (27.8146 - 15.8488 * t))
    m1 = 0.0059 + t * (-7.507 + t * (20.0179 - 10.0282 * t))
    m2 = -5.2673 + t * (15.7311 - 11.4394 * t)
    s1 = exp(-0.8622 - 1.3325 * t)
    s2 = exp(-0.4009 - 1.9894 * t)
    return Mixture([1 - pw, pw], [Normal(m1, s1), Normal(m2, s2)])
