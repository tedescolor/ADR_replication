from adr.dist import Normal, exp


def model(x):
    t = x[0]
    m = -0.073891+t*(-6.255+t*(17.917+t*(-10.753)))
    s = exp(-1.1717+t*(6.6191+t*(-39.783+t*(67.914+t*(-33.164)))))
    return Normal(m, s)
