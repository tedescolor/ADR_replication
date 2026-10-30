from adr.dist import Normal, exp


def model(x):
    t = x[0]
    m = 0.20963 + t * (-12.886 + t * (45.579 + t * (-50.791 + t * 19.098)))
    s = exp(-0.73347 + t * (-3.4719 + t * 5.0241))
    return Normal(m, s)
