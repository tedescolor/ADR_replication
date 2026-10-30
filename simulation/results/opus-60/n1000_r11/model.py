from adr.dist import Normal, Mixture, exp


def model(x):
    """Two equal-weight Gaussian branches with quartic means and a shared log-quadratic scale."""
    t = x[0]
    m1 = 4.466141626686456 * t**4 - 19.675909047991844 * t**3 + 26.055226263960154 * t**2 - 8.092315583774829 * t - 0.254357314271546
    m2 = 10.817149475575647 * t**4 - 36.63629515052805 * t**3 + 35.258022380699586 * t**2 - 10.755475750153396 * t + 0.36387848513259524
    s = exp(-1.73691623233977 * t**2 + 0.6697520655439342 * t - 1.3348967959157634)
    return Mixture([0.5, 0.5], [Normal(m1, s), Normal(m2, s)])
