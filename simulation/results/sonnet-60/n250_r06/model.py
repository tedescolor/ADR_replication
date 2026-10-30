from adr.dist import Normal, exp


def model(x):
    t = x[0]
    m = -0.0083 + -8.0418 * t + 23.3637 * t * t + -14.8124 * t * t * t
    s = exp(-0.7807 + -3.1084 * t + 4.6168 * t * t)
    return Normal(m, s)
