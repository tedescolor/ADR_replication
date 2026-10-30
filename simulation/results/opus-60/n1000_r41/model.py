from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0] - 0.5
    m1 = (((-10.22341 * t + 4.89791) * t + 5.11997) * t + -0.00916)
    m2 = (((-10.1659 * t + -2.36959) * t + 1.84249) * t + -0.0757)
    s1 = exp(-1.28406 * t + -1.48938)
    s2 = exp(-1.23194 * t + -1.66028)
    w = sigmoid(-1.77254 * t + 0.58021)
    return Mixture([w, 1 - w], [Normal(m1, s1), Normal(m2, s2)])
