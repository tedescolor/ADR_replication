import math
from adr.dist import Normal


def model(x):
    x0 = x[0]
    mu = ((-10.221 * x0 + 17.188) * x0 + -5.6838) * x0 + -0.2178
    ls = (4.2816 * x0 + -2.787) * x0 + -0.77333
    return Normal(mu, math.exp(ls))
