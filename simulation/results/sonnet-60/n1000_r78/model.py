from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    x = x[0]
    p = sigmoid((2.1907 + -7.8585*x**1 + 5.9815*x**2))
    return Mixture([p, 1 - p], [Normal((0.10182 + -10.635*x**1 + 36.36*x**2 + -39.569*x**3 + 15.022*x**4 + -2.2769*x**5), exp((-0.97531 + -1.3242*x**1))), Normal((-0.078543 + -6.2084*x**1 + 17.098*x**2 + -8.2706*x**3), exp((-0.69508 + -1.5394*x**1)))])
