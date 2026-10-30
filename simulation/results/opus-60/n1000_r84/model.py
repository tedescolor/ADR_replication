from adr.dist import Normal, Mixture, exp, sqrt, maximum, sigmoid


def model(x):
    """Two-curve Gaussian mixture: quartic means, shared sd = s0*sqrt(1 + k*x)."""
    t = x[0]
    m1 = -0.12383 + t * (-8.66709 + t * (26.13989 + t * (-18.85371 + t * 4.03068)))
    m2 = 0.22844 + t * (-9.28482 + t * (31.96705 + t * (-34.35047 + t * 10.50738)))
    s = exp(-1.22336) * sqrt(maximum(1 + -0.90993 * t, 0.0001))
    w = sigmoid(0.054)
    return Mixture([w, 1 - w], [Normal(m1, s), Normal(m2, s)])
