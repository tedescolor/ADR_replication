from adr.dist import Normal, Mixture, sin, exp, log, sigmoid


def model(x):
    t = x[0]
    # lower branch: linear trend plus sine wave
    m1 = 0.73194995 - 2.26682469 * t + 1.09001205 * sin(4.8455084 * t - 2.23673678)
    # upper branch splits off from the lower one around t = 0.67 (smooth hinge)
    s = 0.10745715
    m2 = m1 + 10.34818475 * s * log(1.0 + exp((t - 0.66666003) / s))
    sd1 = exp(-0.56044703 - 1.64600326 * t)
    sd2 = exp(-1.07335386 - 1.05238927 * t)
    w2 = sigmoid(0.67536028 - 1.00376699 * t)
    return Mixture([1.0 - w2, w2], [Normal(m1, sd1), Normal(m2, sd2)])
