from adr.dist import Normal, Mixture, exp, softmax


def model(x):
    z = 2.0 * x[0] - 1.0
    m0 = (((((0.62328) * z + 0.51244) * z + -2.72578) * z + -0.79902) * z + 1.57447) * z + -0.15437
    s0 = exp((-0.41589) * z + -1.63243)
    m1 = (((((-0.9332) * z + -0.01882) * z + 0.47454) * z + 0.51134) * z + 2.13824) * z + 0.2472
    s1 = exp((0.36037) * z + -2.25264)
    w = softmax(0.0, (1.34432) * z + -0.87652)
    return Mixture(w, [Normal(m0, s0), Normal(m1, s1)])
