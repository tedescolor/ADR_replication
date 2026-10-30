from adr.dist import Normal, Mixture, sin, exp, sigmoid


def model(x):
    t = x[0]
    wb = sigmoid(3.6619 * (t - 0.8461))
    a = Normal(-0.2928 + 0.5459 * sin(7.6039 * t + -3.4108), exp(-0.8802 + -1.0828 * t))
    c = Normal(-1.7619 + t * (3.6684 + 0.7247 * t), exp(-1.1308 + -0.9452 * t))
    return Mixture([1 - wb, wb], [a, c])
