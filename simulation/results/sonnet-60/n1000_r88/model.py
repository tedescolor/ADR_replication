from adr.dist import Normal, Mixture, exp


def model(x):
    t = x[0]
    m1 = 0.076 + t * (-8.081 + t * (21.205 - 10.722 * t))
    m2 = -5.384 * t + 17.152 * t * t - 12.623 * t * t * t - 0.202
    s = exp(-0.928 - 1.357 * t)
    return Mixture([0.528, 0.472], [Normal(m1, s), Normal(m2, 1.074 * s)])
