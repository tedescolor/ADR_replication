from adr.dist import Normal, Mixture, exp, maximum, sigmoid


def model(x):
    """Trunk cubic; past a split point a lower branch peels off (2-component mixture)."""
    t = x[0]
    ma = 0.0830660 + t * (-8.2247096 + t * (21.4981663 - 10.8777747 * t))
    r = maximum(t - 0.5074479, 0.0)
    mb = ma + r * (-4.1634318 - 5.7209837 * r)
    s = exp(-0.8134395 - 1.4040307 * t)
    w = sigmoid(-0.0897818)
    return Mixture([w, 1 - w], [Normal(ma, s), Normal(mb, s)])
