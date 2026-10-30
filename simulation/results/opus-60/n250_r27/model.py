from adr.dist import Normal, Mixture, exp


def model(x):
    t = x[0]
    up = -0.0603530200 + t * (-5.0292240927 + t * (13.9305640128 + t * -6.2606860273))
    lo = 0.0237857374 + t * (-13.0282372194 + t * (46.2503265633 + t * (-50.7534747285 + t * 16.5182405069)))
    s = exp(-1.2448136533 - 0.8401901764 * t)
    return Mixture([0.5, 0.5], [Normal(up, s), Normal(lo, s)])
