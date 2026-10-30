from adr.dist import Normal, Mixture, exp, sigmoid, maximum


def model(x):
    """Pitchfork: one noisy branch, a second branch splits off below it after x = c."""
    x = x[0]
    m = -0.16339 + -5.91239*x + 16.42487*x*x + -7.77183*x*x*x
    u = maximum(x - 0.52677, 0.0)
    s = exp(-1.21292 + 0.60145*x + -1.9776*x*x)
    w = 0.5 * sigmoid(133.7309 * (x - 0.52677))
    return Mixture([1 - w, w], [Normal(m, s), Normal(m - 3.52845*u - 8.61168*u*u, s)])
