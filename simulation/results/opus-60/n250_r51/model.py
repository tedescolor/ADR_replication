from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    """Two-branch Normal mixture in x0: a curved lower branch and a steep linear upper branch
    that switches on (sharp logistic gate) above x0 ~ 0.281; log-polynomial scales."""
    z = x[0] - 0.5
    ml = (((-12.944694) * z + -1.466887) * z + 2.357915) * z + -0.217776
    sl = exp((-1.491397) * z + -1.649567)
    mu = (5.082011) * z + 0.102921
    su = exp((-1.574666) * z + -1.532074)
    w = 0.544625 * sigmoid((x[0] - 0.281398) / 0.000132422)
    return Mixture([1.0 - w, w], [Normal(ml, sl), Normal(mu, su)])
