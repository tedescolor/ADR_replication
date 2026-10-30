from adr.dist import Mixture, Normal, LogNormal


def model(x):
    """Jittered point mass at zero plus a lognormal for positive y; covariates carry no usable signal."""
    return Mixture([0.4495, 0.5505], [Normal(0.0, 0.6308), LogNormal(1.7891, 1.6446)])
