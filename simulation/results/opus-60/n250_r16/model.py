from adr.dist import Normal, Mixture, exp, maximum, minimum


def model(x):
    t = x[0] - 0.5
    mb = -0.018 + t * (2.825 + t * (-6.015 + t * (-15.771 + t * 17.189)))
    sb = exp(-1.326 + t * (-1.498 - 1.844 * t))
    mu = 0.06 + 5.231 * t
    w = minimum(maximum(-0.131 + 0.798 * x[0], 0.001), 0.999)
    return Mixture([1 - w, w], [Normal(mb, sb), Normal(mu, exp(-2.166))])
