from adr.dist import Normal, Mixture, exp, sin, sigmoid


def model(x):
    t = x[0]
    ma = 0.3057 - 1.2225 * t + 0.7922 * sin(5.8137 * t + 3.6347)
    mb = -4.7541 + 11.8502 * t - 4.6167 * t * t
    sa = exp(-0.899 - 1.298 * t)
    sb = exp(0.7305 - 3.223 * t)
    w = 0.5 * sigmoid(16.9793 * (t - 0.586))
    return Mixture([1 - w, w], [Normal(ma, sa), Normal(mb, sb)])
