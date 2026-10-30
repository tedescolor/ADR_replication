from adr.dist import Normal, exp


def model(x):
    t = x[0]
    mu = -0.10604 + t * (-6.33376 + t * (18.9702 + t * -11.9093))
    s = exp(-0.53276 + t * (-4.02048 + t * 5.42847))
    return Normal(mu, s)
