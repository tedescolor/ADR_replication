from adr.dist import Mixture, Normal, exp, sigmoid


def model(x):
    u = x[0] - 0.5
    m1 = -0.080424 + 2.178*u + -3.388*u**2 + -11.385*u**3 + 4.2548*u**4
    m2 = -0.058863 + 5.1465*u + 5.3919*u**2 + -10.934*u**3
    s1 = exp(-1.365 + -0.63592*u + -3.0466*u**2)
    s2 = exp(-1.5036 + -1.3888*u)
    z = 0.031548 + -1.6866*u + 3.0375*u**2
    return Mixture([sigmoid(-z), sigmoid(z)], [Normal(m1, s1), Normal(m2, s2)])
