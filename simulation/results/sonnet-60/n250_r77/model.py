from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = -0.0223 + t * (-10.2248 + t * (37.2702 + t * (-39.757 + t * 11.6848)))
    m2 = -2.3554 + 4.9628 * t
    s1 = exp(-0.8192 + -1.1952 * t)
    s2 = exp(-1.9644)
    w = sigmoid(-5.1363 + 6.5683 * t)
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
