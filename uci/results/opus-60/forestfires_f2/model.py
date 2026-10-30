from adr.dist import Mixture, Normal, LogNormal


def model(x):
    """Blurred point mass at zero (Normal) plus a LogNormal for positive values."""
    return Mixture([0.44636, 0.55364],
                   [Normal(-0.09087, 0.58587), LogNormal(1.67721, 1.69538)])
