from adr.dist import Normal, exp


def model(x):
    t = x[0]
    mu = ((-10.177 * t + 18.271) * t + -6.6955) * t + -0.083322
    ls = (5.1328 * t + -3.7333) * t + -0.69463
    return Normal(mu, exp(ls))
