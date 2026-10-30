from adr.dist import Normal, Mixture, exp, softmax


def model(x):
    t = 2 * x[0] - 1
    m0 = 0.13866 + t * (2.6674 + t * (-0.37617 + t * (-0.019333 + t * (1.8089 + t * (-1.7679)))))
    s0 = exp(-1.5915 + t * (-0.665))
    m1 = -0.20645 + t * (1.2074 + t * (0.26055 + t * (-2.6743 + t * (-0.83165 + t * (1.2397)))))
    s1 = exp(-1.9356 + t * (-0.50928))
    w = softmax(0, -0.10645 + t * (-0.386))
    return Mixture(w, [Normal(m0, s0), Normal(m1, s1)])
