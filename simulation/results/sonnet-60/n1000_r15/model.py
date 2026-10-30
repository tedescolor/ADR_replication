from adr.dist import Normal, Mixture, sigmoid, sin, exp


def model(x):
    t = x[0]
    wt = sigmoid(-2.6948 + 3.3879 * t)
    lower = Normal(-0.2944 + (-0.428 + -0.1646 * t) * sin(7.4991 * t + -0.2218), exp(-1.0411 + -0.9042 * t))
    upper = Normal(-1.8174 + 3.8053 * t + 0.6514 * t * t, exp(-1.0114 + -1.1289 * t))
    return Mixture([1 - wt, wt], [lower, upper])
