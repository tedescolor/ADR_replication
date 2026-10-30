from adr.dist import Normal, exp


def model(x):
    u = x[0] - 0.5
    m = -0.062458+3.9062*u**1+0.94923*u**2+-14.08*u**3
    s = exp(-1.3765+1.1071*u**1+11.588*u**2+2.0811*u**3+-29.747*u**4)
    return Normal(m, s)
