"""Two-branch mixture: a smooth quintic main curve plus a linear upper branch whose
weight grows logistically with x; log-linear heteroscedastic Normal noise on each."""
from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    ma = ((((31.17493 * t - 78.69462) * t + 55.24634) * t - 5.79443) * t - 2.77121) * t - 0.25
    sa = exp(-0.96515 * t - 1.08911)
    mb = 4.60405 * t - 2.04171
    sb = exp(-2.33744 * t - 0.18566)
    w = sigmoid(5.17581 * t - 4.18531)
    return Mixture([1.0 - w, w], [Normal(ma, sa), Normal(mb, sb)])
