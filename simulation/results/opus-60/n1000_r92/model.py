from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = 2 * x[0] - 1
    s = exp(((-0.354121)*t-0.622939)*t-1.46802)
    w = sigmoid((-1.09484)*t+0.452589)
    a = Normal(((((((-0.614152)*t+0.0618447)*t+0.902787)*t-1.17411)*t+0.819385)*t+2.41193)*t+0.0845984, s)
    b = Normal(((((((-0.761368)*t+0.0652406)*t+2.52334)*t-2.69932)*t-1.84209)*t+1.83717)*t-0.143897, s)
    return Mixture([w, 1 - w], [a, b])
