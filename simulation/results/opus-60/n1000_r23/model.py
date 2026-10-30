"""Two-component Gaussian mixture in y with polynomial (in x0) means, log-sds and logit weight."""
from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    mu1 = 0.054418 + t * (-7.756153 + t * (20.44163 + t * (-10.240103)))
    mu2 = -5.032368 + t * (15.293535 + t * (-11.243526))
    sd1 = exp(-0.882092 + t * (-1.285803))
    sd2 = exp(-0.350094 + t * (-1.888598))
    w = sigmoid(10.535555 + t * (-26.915872 + t * (16.837018)))
    return Mixture([w, 1.0 - w], [Normal(mu1, sd1), Normal(mu2, sd2)])
