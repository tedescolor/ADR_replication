from adr.dist import Normal, Mixture, exp, sigmoid, maximum


def model(x):
    t = 2.0 * x[0] - 1.0
    r = maximum(x[0] - 0.461717, 0.0)
    m1 = (((-1.30462) * t + 1.21754) * t + 2.54186) * t + 0.0133716
    m2 = m1 - r * r * (((12.345) * t + -26.5791) * t + 26.4001)
    ls = ((-0.284435) * t + -0.668648) * t + -1.46778
    w = sigmoid(0.135331)
    return Mixture([w, 1.0 - w], [Normal(m1, exp(ls)), Normal(m2, exp(ls))])
