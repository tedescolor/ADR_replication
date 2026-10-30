from adr.dist import Normal, Mixture, exp


def model(x):
    """x < 0.55: one noisy curve; x >= 0.55: two branches (upper, lower) with a shared scale."""
    t = x[0]
    if t < 0.55:
        mean = -0.0425063 + t * (-5.13863 + t * (-1.97343 + t * (56.9429 + t * -64.1765)))
        return Normal(mean, exp(-0.86537 + -1.22856 * t))
    sd = exp(-0.313053 + -2.03149 * t)
    upper = Normal(-4.94663 + t * (11.5092 + t * -4.0434), sd)
    lower = Normal(-3.82614 + t * (12.1617 + t * -9.27742), sd)
    return Mixture([0.514398, 0.485602], [upper, lower])
