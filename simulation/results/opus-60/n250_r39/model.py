from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m_low = -0.0025 + t * (-8.4045 + t * (28.84022 + t * (-28.2986 + t * 6.81666)))
    s_low = exp(-0.94031 + -1.21516 * t)
    m_up = -2.72331 + 5.45781 * t
    s_up = exp(-1.8666)
    w = sigmoid(41.53048 * (t - 0.25872)) * sigmoid(-0.45939)
    return Mixture([1.0 - w, w], [Normal(m_low, s_low), Normal(m_up, s_up)])
