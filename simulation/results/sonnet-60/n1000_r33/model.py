from adr.dist import Mixture, Normal, exp, sigmoid


def model(x):
    t = x[0]
    m1 = ((((-0.13388) * t + -9.7291) * t + 19.94) * t + -7.5922) * t + 0.03597
    m2 = ((-10.788) * t + 14.446) * t + -4.6265
    w = sigmoid(((-16.216) * t + 26.128) * t + -10.281)
    s1 = exp((-1.4196) * t + -0.80896)
    s2 = exp((-2.5107) * t + 0.050996)
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
