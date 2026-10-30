from adr.dist import Normal, Mixture, sigmoid, exp


def model(x):
    t = x[0]
    s = sigmoid(8.6111 * (t - 0.6443))
    m_a = -0.0861 + t * (-7.4612 + t * (20.1717 - 10.1028 * t))
    sd_a = exp(-1.1495 - 1.0179 * s)
    m_b = -4.3182 + t * (13.4705 - 10.1345 * t)
    w = sigmoid(0.6121) * s
    return Mixture([1 - w, w], [Normal(m_a, sd_a), Normal(m_b, exp(-1.8508))])
