from adr.dist import Normal, Mixture, exp, maximum


def model(x):
    t = 2 * x[0] - 1
    upper = -0.00087 + 2.39792*t + 1.18303*t**2 + -0.83895*t**3 + 0.09899*t**4 + -0.36653*t**5
    lower = upper - maximum(x[0] - 0.51948, 0) ** 0.92706 * (3.06679 + 3.82888*t)
    s = exp(-1.55247 + -0.691*t)
    return Mixture([0.5, 0.5], [Normal(upper, s), Normal(lower, s)])
