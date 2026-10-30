import math
from adr.dist import Normal


def model(x):
    u = x[0]
    mu = ((-12.222 * u + 19.575) * u + -6.5317) * u + -0.17793
    ls = (4.0021 * u + -2.39) * u + -0.92395
    return Normal(mu, math.exp(ls))
