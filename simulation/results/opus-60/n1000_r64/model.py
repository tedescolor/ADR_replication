from adr.dist import Mixture, Normal, maximum


def model(x):
    t = 2.0 * x[0] - 1.0
    ma = -0.03994 + t * (2.8418 + t * (0.50389 + t * (-1.82879 + t * (2.01408 + t * (0.1848 + t * (-1.21213))))))
    mb = -0.08384 + t * (1.29846 + t * (-1.19427 + t * (-2.14427 + t * (1.73826 + t * (0.3285 + t * (-0.96282))))))
    s = maximum(0.22639 + t * (-0.13908), 0.001)
    return Mixture([0.5, 0.5], [Normal(ma, s), Normal(mb, s)])
