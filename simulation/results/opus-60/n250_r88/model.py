from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m_a = (-0.0364886) + (-7.48694)*t**1 + (26.4039)*t**2 + (-26.2561)*t**3 + (6.35313)*t**4
    s_a = exp((-0.792359) + (-1.34955)*t**1)
    m_b = (-3.79678) + (9.0654)*t**1 + (-2.73987)*t**2
    w = sigmoid(-0.134579) * sigmoid(10000 * (t - 0.338457))
    return Mixture([1 - w, w], [Normal(m_a, s_a), Normal(m_b, 0.146092)])
