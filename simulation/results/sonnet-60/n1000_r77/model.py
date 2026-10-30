from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    x0 = x[0]
    pu = sigmoid(-3.3906 + 4.0664 * x0)
    lo = Normal((-111.69)*x0**6+(355.92)*x0**5+(-422.59)*x0**4+(213.15)*x0**3+(-32.984)*x0**2+(-2.8134)*x0**1+(-0.037257)*x0**0, exp(-0.88509 + -1.2484 * x0))
    up = Normal(-2.1272 + 4.7228 * x0, exp(-0.64701 + -1.5638 * x0))
    return Mixture([1 - pu, pu], [lo, up])
