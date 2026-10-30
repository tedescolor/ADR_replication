from adr.dist import Normal, Mixture, sin, exp, sigmoid


def model(x):
    t = x[0]
    w = sigmoid(3.4064 * t - 2.9416)
    main = Normal(0.6854 * sin(6.2123 * t - 2.8244) + 0.188 - 0.9942 * t, exp(-0.871 - 1.2771 * t))
    upper = Normal(4.8104 * t - 2.209, exp(-0.3141 - 2.019 * t))
    return Mixture([1 - w, w], [main, upper])
