from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    """Two crossing branches: 2-component normal mixture, polynomial means/log-scales in t=2x-1."""
    t = 2 * x[0] - 1
    w = sigmoid(-0.06802)
    c1 = Normal(((((((-1.39788) * t + -0.0282) * t + 2.69557) * t + -1.94049) * t + -1.59788) * t + 1.28045) * t + -0.08947, exp(((-0.55023) * t + -0.62678) * t + -1.32582))
    c2 = Normal((((-1.18946) * t + 1.11808) * t + 2.55578) * t + 0.00658, exp((-0.58948) * t + -1.66305))
    return Mixture([1 - w, w], [c1, c2])
