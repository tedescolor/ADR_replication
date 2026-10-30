from adr.dist import Normal, Mixture, exp, softmax


def model(x):
    t = x[0] - 0.5
    m0 = -0.2303 + 3.3909 * t - 4.4546 * t ** 2 - 11.8789 * t ** 3
    s0 = exp(-1.5497 + 2.9862 * t - 13.4737 * t ** 2)
    m1 = 0.0497 + 4.9377 * t + 4.8089 * t ** 2 - 9.7305 * t ** 3
    s1 = exp(-1.5008 - 1.1499 * t + 0.2453 * t ** 2)
    z1 = 0.2637 - 3.2603 * t + 8.4423 * t ** 2
    w = softmax(0.0, z1)
    return Mixture(w, [Normal(m0, s0), Normal(m1, s1)])
