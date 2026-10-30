from adr.dist import Normal, Mixture, exp


def model(x):
    x0 = x[0]
    m1 = ((((((-111.5991)*x0+353.1122)*x0+-414.0186)*x0+203.3692)*x0+-28.3344)*x0+-3.5186)*x0+-0.0553
    m2 = -2.1727 + 4.7693 * x0
    s1 = exp(-0.8035 + -1.2831 * x0)
    s2 = exp(-0.9972 + -1.1779 * x0)
    z = exp(-0.6207 + 3.8482 * (x0 - 0.7))
    return Mixture([1.0, z], [Normal(m1, s1), Normal(m2, s2)])
