from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    z = 2 * x[0] - 1
    m1 = -0.0452 + z * (1.0313 + z * (-1.1423 + z * (-1.0863 + z * (0.582 + z * -0.3164))))
    m2 = -0.0549 + z * (2.49 + z * (1.5136 + z * (-1.2007 + z * (-0.2165 + z * -0.051))))
    s1 = exp(-1.4338 - 0.9323 * z)
    s2 = exp(-1.5802 - 0.5574 * z)
    w = sigmoid(-0.3699 + z * (2.111 + z * -1.7878))
    return Mixture([w, 1 - w], [Normal(m1, s1), Normal(m2, s2)])
