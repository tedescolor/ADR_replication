from adr.dist import Normal, Mixture, exp, sin, sigmoid


def model(x):
    t = x[0]
    m1 = -0.132 + 0.569 * sin(6.9021 * t - 3.0847) - 0.3222 * t
    m2 = -2.1755 + 4.7825 * t
    s1 = exp(-0.9031 - 1.2387 * t)
    s2 = exp(-2.0556)
    w = sigmoid(-3.3345 + 3.612 * t)
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
