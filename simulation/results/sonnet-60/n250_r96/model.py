from adr.dist import Normal, exp


def model(x):
    t = x[0]
    m = (((1.04635364 * t - 17.21829315) * t + 25.83789831) * t - 9.27427352) * t + 0.19875178
    s = exp(-0.72698185 + t * (-3.46103431 + 4.98294423 * t))
    return Normal(m, s)
