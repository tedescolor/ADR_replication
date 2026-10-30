from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    z = 2 * x[0] - 1
    w = sigmoid(-0.8747 + 1.4577 * z)
    a = Normal(-0.0015 + z * (2.5609 + z * (1.2672 - 1.2896 * z)), exp(-1.7174 - 0.7533 * z))
    b = Normal(-0.1434 + z * (1.3306 + z * (-1.0169 - 1.218 * z)), exp(-1.4391 - 0.9302 * z))
    return Mixture([1 - w, w], [a, b])
