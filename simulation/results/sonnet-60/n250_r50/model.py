from adr.dist import Normal, exp


def model(x):
    t = x[0]
    m = -0.18293481 + t * (-6.89517326 + t * (20.2203917 + t * -12.18704463))
    s = exp(-0.74777906 + t * (-3.49535573 + t * 5.01117372))
    return Normal(m, s)
