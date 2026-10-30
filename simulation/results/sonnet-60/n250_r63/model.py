from adr.dist import Mixture, Normal, exp, sigmoid


def model(x):
    t = x[0]
    m1 = (((-7.78526194 * t + 5.27630677) * t + 10.29263035) * t - 5.29907167) * t - 0.12609613
    m2 = (-8.10101896 * t + 10.38649449) * t - 3.16371841
    s1 = exp(-0.90694189 - 1.45704094 * t)
    s2 = exp(-0.51089754 - 1.83912211 * t)
    w = sigmoid(4.68742122 * (t - 0.73080285))
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
