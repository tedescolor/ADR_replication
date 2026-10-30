from adr.dist import LogNormal, Mixture, log, where


def model(x):
    """Log-linear median with x2 as a 3-level factor; lognormal with a 3% wide (4x sd) component."""
    mu = (1.8815 - 0.386 * where(x[2] > 150, 1.0, 0.0) - 0.5447 * where(x[2] > 250, 1.0, 0.0)
          - 1.0674 * log(x[1]) + 0.9931 * log(x[4]))
    return Mixture([0.97, 0.03], [LogNormal(mu, 0.2954), LogNormal(mu, 1.1816)])
