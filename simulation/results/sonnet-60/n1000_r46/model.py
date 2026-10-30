from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    x0 = x[0]
    m1 = -5.043*x0**4 + 0.964805*x0**3 + 12.3163*x0**2 + -5.70894*x0 + -0.0627183
    m2 = -9.03035*x0**2 + 11.7776*x0 + -3.66743
    s1 = exp(-0.856819 + -1.30993*x0)
    s2 = exp(-0.62993 + -1.65743*x0)
    p2 = sigmoid(-3.43383 + 4.3145*x0)
    return Mixture([1 - p2, p2], [Normal(m1, s1), Normal(m2, s2)])
