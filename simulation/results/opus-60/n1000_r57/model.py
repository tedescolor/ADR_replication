from adr.dist import Normal, Mixture, exp, log, sigmoid, maximum, abs


def model(x):
    t = 2.0 * x[0] - 1.0
    z = (x[0] - 0.509103) / 0.0127477
    r = 0.0127477 * (maximum(z, 0.0) + log(1.0 + exp(-abs(z))))
    b = ((((0.112201) * t + -1.41444) * t + 0.753551) * t + 2.27801) * t + 0.00068147
    w = sigmoid(0.120646)
    m1 = b + r * ((1.28026) * t + 0.346115)
    m2 = b + r * ((-2.3668) * t + -3.1901)
    s1 = exp(((-0.425133) * t + -0.815533) * t + -1.30555)
    s2 = exp(((0.0261861) * t + -0.62425) * t + -1.54212)
    return Mixture([w, 1.0 - w], [Normal(m1, s1), Normal(m2, s2)])
