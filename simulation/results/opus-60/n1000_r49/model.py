from adr.dist import Normal, Mixture, exp, maximum, sigmoid


def model(x):
    t = x[0]
    m1 = -0.0753228 + t * (-5.10315 + t * (-8.73857 + t * (122.545 + t * (-265.137 + t * (226.226 + t * -70.7019)))))
    m2 = m1 + 9.18146 * maximum(t - 0.442167, 0.0) ** exp(0.497762)
    s = exp(-1.07457 + t * (-0.156305 + t * -1.18718))
    w = sigmoid(-0.0617742)
    return Mixture([1.0 - w, w], [Normal(m1, s), Normal(m2, s)])
