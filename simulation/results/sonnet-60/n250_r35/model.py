from adr.dist import Normal, exp


def model(x):
    t = x[0]
    mu = ((-13.45 * t + 21.95) * t - 7.725) * t - 0.05127
    s = exp((4.903 * t - 3.416) * t - 0.7192)
    return Normal(mu, s)
