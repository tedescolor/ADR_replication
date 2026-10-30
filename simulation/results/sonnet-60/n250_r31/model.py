from adr.dist import Normal, exp


def model(x):
    x0 = x[0]
    mu = 0.253 + x0 * (-10.08 + x0 * (27.6712 + x0 * -17.5009))
    ls = -0.7839 + x0 * (-2.9852 + x0 * 4.5302)
    return Normal(mu, exp(ls))
