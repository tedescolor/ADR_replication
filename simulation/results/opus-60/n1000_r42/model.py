from adr.dist import Normal, Mixture, exp, log, sigmoid


def model(x):
    """Two-branch Normal mixture: the lower branch splits off the upper one near x=0.544."""
    t = x[0]
    m1 = -0.06173 + t * (-6.29421 + t * (14.26501 + t * (-1.1814 + t * -4.24751)))
    k = exp(-3.97188)
    h = k * log(1.0 + exp((t - 0.5444) / k))
    m2 = m1 - h * (5.4373 + 4.76584 * h)
    s = exp(-1.01523 + t * (0.03735 - 1.52719 * t))
    w = sigmoid(0.30898)
    return Mixture([w, 1.0 - w], [Normal(m1, s), Normal(m2, s)])
