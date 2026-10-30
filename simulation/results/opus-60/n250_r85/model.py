from adr.dist import Normal, Mixture, sigmoid, exp


def model(x):
    t = x[0]
    c = t - 0.5
    m1 = -0.0295 + c * (2.9888 + c * (-5.9657 + c * (-16.585 + c * 18.4835)))
    s1 = exp(-1.4543 - 1.2375 * c)
    m2 = 1.6534 + 4.8914 * (t - 0.8)
    s2 = exp(-1.9653)
    w = sigmoid(39.6725 * (t - 0.6236)) * sigmoid(-0.3172)
    return Mixture([1.0 - w, w], [Normal(m1, s1), Normal(m2, s2)])
