from adr.dist import Normal, Mixture, sin, exp, sigmoid


def model(x):
    t = x[0]
    # lower branch: linear trend + damped sine, log-linear sd
    m1 = 1.2588 - 2.806 * t + exp(0.5057 - 0.5974 * t) * sin(4.8902 * t - 2.4084)
    s1 = exp(-0.8511 - 1.4434 * t)
    # upper branch: straight line, constant sd, switched on for t > ~0.55
    m2 = -1.9816 + 4.5805 * t
    s2 = exp(-1.8144)
    w = sigmoid(-18.628 + 45.0866 * t - 27.0606 * t * t)
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
