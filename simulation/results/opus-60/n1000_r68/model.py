from adr.dist import Normal, Mixture, abs, sigmoid


def model(x):
    t = x[0]
    m1 = ((-10.9048 * t + 21.8086) * t - 8.4888) * t + 0.1178
    m2 = ((-10.1125 * t + 12.5905) * t - 3.0175) * t - 0.4663
    s = abs(0.3642 - 0.2756 * t) + 0.001
    w = sigmoid(1.6393 - 2.0527 * t)
    return Mixture([w, 1 - w], [Normal(m1, s), Normal(m2, s)])
