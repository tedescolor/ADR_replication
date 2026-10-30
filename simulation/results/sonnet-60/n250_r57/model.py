from adr.dist import Normal, exp


def model(x):
    t = x[0]
    mu = 0.08632777 + t * (-9.09331055 + t * (28.90138434 + t * (-26.39202459 + t * 7.58453543)))
    s = exp(-0.70528873 + t * (-3.18683628 + t * 4.62269726))
    return Normal(mu, s)
