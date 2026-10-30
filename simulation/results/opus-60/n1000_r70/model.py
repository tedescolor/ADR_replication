"""Two-branch Gaussian mixture: one curve for x0 < c, which splits into two equally
weighted branches above c (upper = lower + cubic in (x0 - c)); shared log-quadratic sd."""
from adr.dist import Normal, Mixture, maximum, exp


def model(x):
    t = 2.0 * x[0] - 1.0
    m1 = ((((((((3.0778)*t+-0.50289)*t+-6.83164)*t+1.18468)*t+5.51801)*t+-2.6159)*t+-2.14277)*t+1.42963)*t+-0.0691837
    u = maximum(x[0] - 0.503387, 0.0)
    m2 = m1 + ((-7.7588 * u + 12.1108) * u + 2.87094) * u
    s = exp((-0.410545 * t + -0.640397) * t + -1.4205)
    return Mixture([0.5, 0.5], [Normal(m1, s), Normal(m2, s)])
