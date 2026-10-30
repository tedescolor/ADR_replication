from adr.dist import Normal, Mixture, sin, exp, maximum


def model(x):
    t = x[0]
    m1 = 0.576101 - 1.912167 * t + 1.000815 * sin(5.015063 * t - 2.268477)
    h = maximum(t - 0.359125, 0.0)
    m2 = m1 + 8.501910 * h * h
    s = exp(-0.972115 + t * (-0.561298 - 0.811692 * t))
    w = 0.468807
    return Mixture([1 - w, w], [Normal(m1, s), Normal(m2, s)])
