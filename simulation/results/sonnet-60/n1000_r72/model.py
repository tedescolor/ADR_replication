from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    mA = (((13.90417 * t - 43.28910) * t + 39.30872) * t - 11.00870) * t + 0.113119
    mB = 4.755566 * t - 2.166579
    sA = exp(-0.818873 - 1.404780 * t)
    sB = exp(-0.943065 - 1.204950 * t)
    w = sigmoid(-3.267223 + 3.987554 * t)
    return Mixture([1 - w, w], [Normal(mA, sA), Normal(mB, sB)])
