from adr.dist import Mixture, Normal, sin, exp, sigmoid


def model(x):
    t = x[0]
    w = sigmoid(-3.906 + 4.626 * t)
    lower = Normal(-0.275 - 0.543 * sin(7.628 * t - 0.301), exp(-0.918 - 1.05 * t))
    upper = Normal(-2.202 + 4.8 * t, exp(-1.07 - 1.125 * t))
    return Mixture([1 - w, w], [lower, upper])
