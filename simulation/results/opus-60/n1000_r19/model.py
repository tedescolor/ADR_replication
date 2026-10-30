from adr.dist import Normal, Mixture, sigmoid, exp, maximum


def model(x):
    t = x[0]
    base = ((-12.735893 * t + 21.998731) * t - 7.977224) * t + 0.043711
    r = maximum(t - 0.498028, 0.0)
    m_up = base + (4.287321 * r + 0.226679) * r
    m_lo = base + (-2.463985 * r - 3.313825) * r
    s_up = exp((-1.605291 * t + 0.519790) * t - 1.153160)
    s_lo = exp((-0.463223 * t - 0.972172) * t - 0.879573)
    w = sigmoid(0.146059)
    return Mixture([w, 1 - w], [Normal(m_up, s_up), Normal(m_lo, s_lo)])
