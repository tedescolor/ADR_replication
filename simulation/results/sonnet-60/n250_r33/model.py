from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    mA = (((-9.4768) * t + 18.9448) * t + -6.8474) * t + -0.1056
    lsA = (-0.5795) * t + -1.3302
    mB = ((-13.2598) * t + 18.5841) * t + -6.3431
    lsB = (-1.4981) * t + -0.751
    z = ((18.9963) * t + -32.19) * t + 13.2713
    w = sigmoid(z)
    return Mixture([w, 1 - w], [Normal(mA, exp(lsA)), Normal(mB, exp(lsB))])
