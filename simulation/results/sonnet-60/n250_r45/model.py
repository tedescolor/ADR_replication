from adr.dist import Normal, exp


def model(x):
    t = x[0]
    mu = 0.12922 + t * (-9.2652 + t * (25.823 + t * -16.334))
    s = exp(-0.6782 + t * (-3.8593 + t * 5.3505))
    return Normal(mu, s)
