from adr.dist import Normal, Mixture, sigmoid, exp


def model(x):
    t = x[0]
    m1 = 0.13157 + t * (-8.3856 + t * (21.91918 + t * -11.26057))
    s1 = exp(-0.67337 - 1.70182 * t)
    m2 = -0.34547 + t * (-3.2461 + t * (12.56602 + t * -9.99575))
    s2 = exp(-1.12219 - 0.81413 * t)
    w = sigmoid(0.12807)
    return Mixture([w, 1 - w], [Normal(m1, s1), Normal(m2, s2)])
