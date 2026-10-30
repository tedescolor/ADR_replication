from adr.dist import Normal, Mixture, sigmoid, exp


def model(x):
    t = x[0]
    mA = ((-10.4013 * t + 21.0453) * t - 8.1041) * t + 0.0353
    mB = (-16.1204 * t + 22.9438) * t - 7.9798
    s = exp(-1.4153 * t - 0.7904)
    w = 0.4695 * sigmoid(25600.0 * (t - 0.5619))
    return Mixture([1.0 - w, w], [Normal(mA, s), Normal(mB, s)])
