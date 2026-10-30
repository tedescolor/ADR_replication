from adr.dist import Normal, exp


def model(x):
    u = x[0]
    mu = -0.15114 + -5.9035*u**1 + 17.096*u**2 + -10.009*u**3
    sd = exp(-0.61114 + -3.7932*u**1 + 5.2056*u**2)
    return Normal(mu, sd)
