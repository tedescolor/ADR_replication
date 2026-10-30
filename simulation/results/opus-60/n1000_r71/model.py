from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    """Two-branch Gaussian mixture: cubic means, log-polynomial scales, logistic weight."""
    z = 2.0 * x[0] - 1.0
    w = sigmoid(((2.97014)*z + -3.92254)*z + 1.32862)
    a = Normal((((-1.15261)*z + 1.19265)*z + 2.45446)*z + 0.02577, exp((-0.66192)*z + -1.45847))
    b = Normal((((-0.94674)*z + -1.28626)*z + 1.39229)*z + -0.14686, exp((-0.96234)*z + -1.44889))
    return Mixture([w, 1.0 - w], [a, b])
