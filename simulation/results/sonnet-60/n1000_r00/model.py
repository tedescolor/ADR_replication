from adr.dist import Mixture, Normal, exp, sigmoid


def model(x):
    x = x[0]
    m1 = -0.03212 + -7.489*x + 20.1*x**2 + -10.07*x**3
    s1 = exp(-1.046 + -0.299*x + -1.082*x**2)
    m2 = -5.097 + 15.55*x + -11.43*x**2
    s2 = exp(-0.00538 + -2.381*x)
    w = sigmoid(-12.19 + 30.21*x + -18.47*x**2)
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
