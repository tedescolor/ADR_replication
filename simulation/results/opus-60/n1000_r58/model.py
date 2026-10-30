from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    """Two-branch Gaussian mixture: main curve A (quartic) plus lower branch B (cubic)
    that appears after x ~ 0.48; shared log-linear noise scale."""
    t = x[0] - 0.5
    mA = -0.05534283 + t * (5.025559 + t * (6.04165569 + t * (-10.09563176 - 3.77681665 * t)))
    sA = exp(-1.5307643 - 1.44223525 * t)
    mB = 0.03676126 + t * (1.05097959 + t * (-0.39946681 - 11.78470072 * t))
    sB = sA * exp(0.09112466)
    pB = sigmoid(0.17562291) * sigmoid((x[0] - 0.48256033) * exp(2.63669176))
    return Mixture([1 - pB, pB], [Normal(mA, sA), Normal(mB, sB)])
