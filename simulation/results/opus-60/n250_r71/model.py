from adr.dist import Normal, Mixture, sin, exp, sigmoid


def model(x):
    t = x[0]
    m1 = 0.3379 - 1.3108 * t + 0.8092 * sin(5.7226 * t + 3.6458)
    m2 = -2.5471 + 5.2497 * t
    w = sigmoid(30.0 * (t - 0.4066)) * sigmoid(-0.345)
    s1 = exp(-0.8497 - 1.3004 * t)
    s2 = exp(-0.6566 - 1.5467 * t)
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
