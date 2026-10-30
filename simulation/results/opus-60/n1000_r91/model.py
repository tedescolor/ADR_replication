from adr.dist import Normal, Mixture, exp, maximum, minimum


def model(x):
    """Two-branch Gaussian mixture with shared noise scale; t = 2*x0 - 1.
    Lower branch: degree-6 polynomial. Upper branch: lower + h*(cubic in h),
    h = max(0, t - t0), so the branches coincide below t0. Constant branch weight."""
    t = minimum(1.0, maximum(-1.0, 2.0 * x[0] - 1.0))
    m_lo = -0.08083 + t * (1.68026 + t * (-1.87853 + t * (-3.08723 + t * (3.46449 + t * (0.88473 + t * (-2.0083))))))
    h = maximum(0.0, t - (-0.15081))
    m_up = m_lo + h * (-0.00346 + h * (3.4937 + h * (-0.75674)))
    s = exp(-1.46214 + t * (-0.68612 + t * (-0.21501)))
    w = 0.44077
    return Mixture([1.0 - w, w], [Normal(m_lo, s), Normal(m_up, s)])
