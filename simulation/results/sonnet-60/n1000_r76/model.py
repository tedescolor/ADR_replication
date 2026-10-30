from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    u = x[0] - 0.5
    ma = -0.11890 + u * (2.6321 + u * (-4.0297 + u * (-14.641 + u * 10.755)))
    mb = 0.15245 + 4.8934 * u
    sa = exp(-1.4820 - 1.2974 * u)
    sb = exp(-1.8678)
    w = sigmoid(-1.3070 + 4.1249 * u)
    return Mixture([1 - w, w], [Normal(ma, sa), Normal(mb, sb)])
