from adr.dist import Gamma, Mixture, exp, log


def model(x):
    """Gamma GLM (log link) with a fixed 10% low-shape Gamma contamination of the same mean."""
    m = exp(1.02212297 - 0.01533269 * x[0] - 0.06030915 * x[1] - 0.00594222 * x[2]
            - 0.0023628 * x[3] + 0.64598455 * log(x[4]))
    a = 12.45813
    return Mixture([0.9, 0.1], [Gamma(a, a / m), Gamma(2.0, 2.0 / m)])
