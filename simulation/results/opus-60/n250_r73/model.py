from adr.dist import Normal, Mixture, exp, maximum, sigmoid


def model(x):
    t = x[0]
    m = 0.1173619 - 8.650472 * t + 23.76367 * t ** 2 - 14.22840 * t ** 3
    r = maximum(t - 0.5103345, 0.0)
    up = m + 0.1844492 * r + 5.696479 * r * r
    lo = m - 3.449932 * r - 1.297732 * r * r
    s = exp(-0.9008296 - 1.315726 * t)
    w = sigmoid(-0.0201224)
    return Mixture([w, 1.0 - w], [Normal(up, s), Normal(lo, s)])
