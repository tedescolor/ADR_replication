from adr.dist import Normal, Mixture, sigmoid, exp


def model(x):
    t = x[0]
    m1 = -0.8545 + t * (-0.7674 + t * (8.7850 - 8.1318 * t))
    m2 = 0.0195 + t * (-8.0318 + t * (21.5261 - 11.0516 * t))
    a = sigmoid(5.1856 + t * (-13.4401 + 8.4559 * t))
    s1 = exp(-0.6147 - 1.6599 * t)
    s2 = exp(-0.8735 - 1.2861 * t)
    return Mixture([1 - a, a], [Normal(m1, s1), Normal(m2, s2)])
