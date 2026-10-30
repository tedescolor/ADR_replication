from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    """Two crossing branches. Main branch: cubic mean rising from ~0 to ~2.5, log-sd
    linear in x0. Second branch (quadratic mean, peaks ~0.1 at x0~0.65), log-sd linear,
    switched on smoothly around x0 ~ 0.22 with weight ~0.39."""
    t = x[0]
    m_a = ((-8.853 * t + 18.166) * t - 6.771) * t - 0.043
    s_a = exp(-1.309 * t - 0.946)
    m_b = (-7.618 * t + 9.776) * t - 3.013
    s_b = exp(-1.408 * t - 0.757)
    w = sigmoid(-0.456) * sigmoid(100.0 * (t - 0.221))
    return Mixture([1 - w, w], [Normal(m_a, s_a), Normal(m_b, s_b)])
