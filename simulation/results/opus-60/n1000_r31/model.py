"""Two-branch Gaussian mixture in t = 2x - 1: polynomial branch means, shared
log-linear sd, cubic-logit weight of the upper branch."""
from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = 2 * x[0] - 1
    m1 = (((((((1.72808) * t + -1.07326) * t + -2.94258) * t + 2.17718) * t + -0.440393) * t + -1.54227) * t + 1.17238) * t + -0.0821699
    m2 = ((-0.668968) * t + 3.22807) * t + -0.0186953
    sd = exp((-0.665954) * t + -1.5412)
    w = sigmoid((((10.6714) * t + -13.2922) * t + 4.04386) * t + -0.178624)
    return Mixture([1 - w, w], [Normal(m1, sd), Normal(m2, sd)])
