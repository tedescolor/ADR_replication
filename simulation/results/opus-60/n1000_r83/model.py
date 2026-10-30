from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    """Two crossing branches: Normal mixture with polynomial means in t = x - 0.5,
    log-polynomial scales and a logistic-quadratic weight for the lower branch."""
    t = x[0] - 0.5
    a0, a1, a2, a3, a4, a5 = 0.00899, 4.81777, 4.59731, -6.45343, 1.94011, -13.54317
    b0, b1, b2, b3 = -0.12545, 2.42448, -4.3224, -8.03894
    ma = a0 + t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
    mb = b0 + t * (b1 + t * (b2 + t * b3))
    sa = exp(-1.45998 + t * (-1.38844 + t * -0.99718))
    sb = exp(-1.39285 + t * -1.98884)
    pb = sigmoid(-1.38632 + t * (7.10349 + t * -7.75405))
    return Mixture([1.0 - pb, pb], [Normal(ma, sa), Normal(mb, sb)])
