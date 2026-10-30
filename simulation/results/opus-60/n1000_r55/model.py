from adr.dist import Normal, Mixture, exp, maximum


def model(x):
    t = x[0]
    s = exp(-0.813 - 1.416 * t)
    a = ((-10.25 * t + 20.237) * t - 7.505) * t - 0.014
    b = a + maximum(t - 0.395, 0) ** 2 * (4.195 * t - 13.653)
    return Mixture([0.5, 0.5], [Normal(a, s), Normal(b, s)])
