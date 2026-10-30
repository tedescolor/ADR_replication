from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = (((13.5004 * t - 40.3224) * t + 35.6103) * t - 9.6146) * t - 0.0160
    s1 = exp(-1.3156 * t - 0.9081)
    m2 = 5.0055 * t - 2.3624
    s2 = exp(-1.9021)
    w = 0.5 * sigmoid(80.0 * (t - 0.4877))
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
