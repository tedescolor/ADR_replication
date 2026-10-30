from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = (((-11.035472) * t + 15.741839) * t + -6.329253) * t + 0.629595
    m2 = (((-9.647662) * t + 19.426623) * t + -7.249844) * t + -0.040766
    s1 = exp(((-8.38055) * t + 10.388103) * t + -4.778141)
    s2 = exp((-1.319119) * t + -0.848944)
    w2 = sigmoid(((10.798419) * t + -17.696146) * t + 7.050573)
    return Mixture([1.0 - w2, w2], [Normal(m1, s1), Normal(m2, s2)])
