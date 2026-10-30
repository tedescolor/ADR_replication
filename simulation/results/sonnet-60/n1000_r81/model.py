from adr.dist import Normal, exp


def model(x):
    t = x[0]
    mu = (((((27.28288) * t + -59.36602) * t + 30.37125) * t + 8.66164) * t + -6.16631) * t + -0.06481
    ls = ((((-22.67974) * t + 45.06755) * t + -23.44674) * t + 2.53201) * t + -1.00545
    return Normal(mu, exp(ls))
