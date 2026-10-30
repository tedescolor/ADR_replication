from adr.dist import Normal, Mixture, exp, log, sigmoid, maximum, abs


def model(x):
    t = x[0]
    c = 0.08823798 + t * (-8.59546919 + t * (26.10587288 + t * (-21.36986668 + t * 4.53704989)))
    h = exp(-2.40265562)
    z = (t - 0.62623687) / h
    u = h * (maximum(z, 0) + log(1 + exp(-abs(z))))
    g = 4.57563457 * u
    s1 = exp(-1.19284193 + t * (-0.10551942 + t * -1.10404469))
    s2 = exp(-0.85526369 + t * (-0.59404561 + t * -0.84700547))
    w = sigmoid(-0.06516092)
    return Mixture([w, 1 - w], [Normal(c + g, s1), Normal(c - g, s2)])
