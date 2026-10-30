from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    z = 2 * x[0] - 1
    c = ((((-1.49991) * z - 2.25629) * z + 1.41678) * z + 2.39045) * z - 0.07983
    m1 = c + exp(((-2.18956) * z + 7.74437) * z - 4.61552)
    m2 = c - exp(((-4.80656) * z + 7.08509) * z - 2.33073)
    s = exp(((-0.55446) * z - 0.70633) * z - 1.32636)
    w = sigmoid(0.31566)
    return Mixture([w, 1 - w], [Normal(m1, s), Normal(m2, s)])
