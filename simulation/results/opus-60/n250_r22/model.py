from adr.dist import Normal, Mixture, exp


def model(x):
    t = x[0]
    c = -0.219
    m1 = c + t * (-6.595 + t * (19.028 - 9.716 * t))
    m2 = c + t * (-4.475 + t * (15.086 - 11.436 * t))
    s = exp(-1.014 - 1.361 * t * t)
    return Mixture([0.595, 0.405], [Normal(m1, s), Normal(m2, s)])
