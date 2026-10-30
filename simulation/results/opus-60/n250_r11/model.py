from adr.dist import Normal, Mixture, sigmoid, exp


def model(x):
    t = x[0]
    m_main = ((-10.8639 * t + 21.3925) * t - 8.1406) * t + 0.0997
    s_main = exp((-1.6143 * t + 0.2337) * t - 1.1865)
    m_low = ((21.3805 * t - 60.6753) * t + 52.5948) * t - 14.1803
    s_low = exp(-1.9559)
    w = sigmoid(80.0 * (t - 0.4921)) * sigmoid(-0.3494)
    return Mixture([1.0 - w, w], [Normal(m_main, s_main), Normal(m_low, s_low)])
