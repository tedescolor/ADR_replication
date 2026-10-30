from adr.dist import Normal, exp


def model(x):
    t = x[0] - 0.5
    mu = 0.000894 + t * (4.25787 + t * (1.169416 + t * -15.743053))
    s = exp(-1.300716 + t * (1.426247 + t * 5.573772))
    return Normal(mu, s)
