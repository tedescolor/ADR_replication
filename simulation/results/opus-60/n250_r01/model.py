from adr.dist import Normal, Mixture, sin, exp, sigmoid


def model(x):
    t = x[0]
    lo = Normal(0.6334 * sin(6.374 * t - 2.881) + 0.0651 - 0.7005 * t, exp(-0.9726 - 1.05 * t))
    up = Normal(-4.1085 + 9.7683 * t - 3.1353 * t * t, exp(-0.0625 - 2.1152 * t))
    w = sigmoid(-0.1454) * sigmoid(60 * (t - 0.5012))
    return Mixture([1 - w, w], [lo, up])
