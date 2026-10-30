from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m = 10.39984802 * t**4 - 35.963685 * t**3 + 34.46247181 * t**2 - 10.02632797 * t + 0.11945818
    s = exp(-0.91424148 - 1.22956427 * t)
    u = 4.77203998 * t - 2.18550581
    w = sigmoid(4.40422826 * (t - 0.86722694))
    return Mixture([1 - w, w], [Normal(m, s), Normal(u, 0.139)])
