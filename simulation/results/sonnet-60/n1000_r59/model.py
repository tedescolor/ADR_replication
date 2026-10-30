from adr.dist import Mixture, Normal, exp, sigmoid


def model(x):
    t = x[0]
    mA = ((((12.66109 * t + -42.69057) * t + 41.00919) * t + -12.0349) * t + 0.07107)
    mB = (((-8.93695 * t + 18.38532) * t + -6.93924) * t + 0.02423)
    sA = exp(-1.32582 * t + -0.94516)
    sB = exp(-1.24555 * t + -0.94475)
    pB = sigmoid(-2.15123 * (t - 0.82687))
    return Mixture([1 - pB, pB], [Normal(mA, sA), Normal(mB, sB)])
