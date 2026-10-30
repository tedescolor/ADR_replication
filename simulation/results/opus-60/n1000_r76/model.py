"""Two-branch Gaussian mixture: a single curve that splits into two branches near x=0.51."""
from adr.dist import Normal, Mixture, exp, log, abs, maximum, sigmoid


def model(x):
    u = 2 * x[0] - 1
    # upper branch mean (also the common curve before the split): cubic in u
    m1 = -0.009375582 + u * (2.447564612 + u * (1.240181898 + u * -1.170239774))
    # lower branch peels off: m2 = m1 - (a r + b r^2), r = softplus split at c
    z = 40 * (x[0] - 0.5105748256)
    r = (maximum(z, 0) + log(1 + exp(-abs(z)))) / 40
    m2 = m1 - r * (3.827732887 + 6.728483633 * r)
    s1 = exp(-1.378325828 + u * (-0.7615799714 + u * -0.1797260719))
    s2 = exp(-1.398395163 + u * (-0.5918964711 + u * -0.4282855002))
    w2 = sigmoid(-0.05964181174)
    return Mixture([1 - w2, w2], [Normal(m1, s1), Normal(m2, s2)])
