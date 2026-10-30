from adr.dist import Normal, Mixture, exp


def model(x):
    """Two equally weighted Gaussian branches with a shared x-dependent scale."""
    t = x[0]
    upper = (((-9.41583) * t + 19.59983) * t + -7.97889) * t + 0.32551
    lower = ((((11.38586) * t + -38.55742) * t + 36.40475) * t + -10.07356) * t + -0.14257
    s = exp(((-1.57861) * t + 0.67035) * t + -1.46601)
    return Mixture([0.5, 0.5], [Normal(upper, s), Normal(lower, s)])
