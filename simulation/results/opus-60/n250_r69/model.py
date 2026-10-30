"""Two-branch Gaussian mixture: a main curve, plus a lower branch that splits off for x0 > ~0.51."""
from adr.dist import Normal, StudentT, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    a = 0.453499 * sigmoid(2389910000.0 * (t - 0.514029))
    main = Normal(0.142284 + t * (-11.4356 + t * (37.7857 + t * (-36.9189 + t * (13.0998)))), exp(-0.802498 + t * (-1.55021)))
    low = Normal(-4.66413 + t * (14.216 + t * (-10.5314)), exp(-0.619186 + t * (-1.51153)))
    return Mixture([1 - a, a], [main, low])
