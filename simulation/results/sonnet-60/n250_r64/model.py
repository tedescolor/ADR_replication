from adr.dist import Normal, exp


def model(x):
    t = x[0]
    m = 0.0074 + t * (-7.3228 + t * (20.1944 + t * -11.7874))
    s = exp(-0.7346 + t * (-3.394 + t * 4.8665))
    return Normal(m, s)
