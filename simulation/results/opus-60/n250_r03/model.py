from adr.dist import Normal, Mixture, exp, log, sigmoid


def model(x):
    """Pitchfork: one cubic branch that splits at x~0.61 into an upper and a lower branch."""
    t = x[0]
    k = 17.6409
    z = k * (t - 0.611701543)
    g = sigmoid(z)
    sp = log(1 + exp(z)) / k
    f = 0.00160405374 + t * (-7.3046144 + t * (19.8471229 - 10.307146 * t))
    m1 = f + 0.644369729 * sp
    m2 = f - 8.31487664 * sp
    s = 0.447901003 - 0.428465216 * t
    base = log(1 + exp(8 * s)) / 8 + 0.001
    s2 = base * exp(-0.138784201 * g)
    w = sigmoid(0.0615826024)
    return Mixture([w, 1 - w], [Normal(m1, base), Normal(m2, s2)])
