from adr.dist import Normal, StudentT, sin, cos, arctan2, maximum, where, pi


def saw(t):
    """Sawtooth in [-0.5, 0.5) with unit period, jumping at integer t."""
    return arctan2(-sin(2 * pi * t), -cos(2 * pi * t)) / (2 * pi)


def model(x):
    z0 = x[0] - 4.5
    g = where(x[2] > 0.085, 1.0, 0.0)
    u = maximum(x[2] - 0.095, 0.0)
    s0 = saw((x[0] - 3.05) / 0.5)
    s2 = saw((x[2] - 0.095) / 0.05) * g
    f = (1.33699 + 0.0496377 * z0 + -0.151492 * x[2] + 0.0529288 * g
         + u * (2.64912 + u * (-5.04481 + 4.8803 * u) + z0 * (0.555304 + -0.97771 * u))
         + s0 * (-0.145908 + -0.0186176 * z0 + -0.13581 * u + 0.230968 * x[2])
         + s2 * (-0.140147 + -0.0175934 * z0 + u * (0.214008 + -0.281212 * u) + 0.00786107 * s0))
    if x[3] < 18.0:
        return StudentT(3, f, 0.0055385)
    return Normal(1.12571 + 0.000952604 * x[3] + 0.0464544 * f, 0.00287453)
