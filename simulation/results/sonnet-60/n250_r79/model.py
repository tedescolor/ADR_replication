from adr.dist import Normal, exp


def model(x):
    t = x[0]
    mu = 0.0201263744 + t * (-7.91656728 + t * (23.0691124 + t * -14.9222413))
    s = exp(-0.61807576 + t * (-4.10059074 + t * 5.61040887))
    return Normal(mu, s)
