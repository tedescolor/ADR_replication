from adr.dist import Normal, Mixture, exp


def model(x):
    t = x[0]
    m1 = 0.1248 + t * (-10.61 + t * (38.63 + t * (-43.91 + t * 14.89)))
    m2 = -2.309 + 4.875 * t
    s1 = exp(-0.9369 - 1.109 * t)
    s2 = exp(-0.8598 - 1.433 * t)
    w = 1.0 / (1.0 + exp(-(3.834 - 5.456 * t)))
    return Mixture([w, 1.0 - w], [Normal(m1, s1), Normal(m2, s2)])
