"""Two-branch mixture: a main curve (degree-6 polynomial mean, log-quadratic scale) plus
an upper branch (quadratic mean, constant scale) whose weight rises smoothly around
x0 ~ 0.44 and is capped below 1/2."""
from adr.dist import Normal, Mixture, exp, sigmoid, minimum, maximum


def model(x):
    z = 2.0 * x[0] - 1.0
    ma = ((((((-3.99063)*z+0.802907)*z+6.46666)*z-2.79012)*z-3.16163)*z+1.47013)*z+0.0527337
    sa = exp(minimum(maximum(((-0.337542)*z-0.692068)*z-1.43035, -4.0), 2.0))
    mb = ((-1.23047)*z+4.00166)*z-0.219491
    sb = 0.140235
    pb = 0.477442 * sigmoid(2.56479 + 20 * z)
    return Mixture([1.0 - pb, pb], [Normal(ma, sa), Normal(mb, sb)])
