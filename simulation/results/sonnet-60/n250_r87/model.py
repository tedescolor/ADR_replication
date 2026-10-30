from adr.dist import Normal, Mixture, sigmoid, sin, exp


def model(x):
    t = x[0]
    w = 0.5 * sigmoid(60.0 * (t - 0.6097))
    lo = Normal(-0.2390 + 0.4867 * sin(7.7861 * t - 3.4711), exp(-1.0020 - 0.8732 * t))
    hi = Normal(-2.0534 + 4.6369 * t, exp(-2.0693))
    return Mixture([1.0 - w, w], [lo, hi])
