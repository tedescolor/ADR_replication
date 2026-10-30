from adr.dist import Mixture, Normal, LogNormal


def model(x):
    w = 0.3364
    return Mixture([w, 1.0 - w], [Normal(-0.2619, 0.4885), LogNormal(1.1803, 1.8063)])
