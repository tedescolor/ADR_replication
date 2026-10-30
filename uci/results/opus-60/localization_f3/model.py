from adr.dist import Gamma, exp, log


def model(x):
    """Gamma GLM with log link: log E[y] linear in log x0, log x1, x2, log x4."""
    mean = exp(3.96683 - 0.29536 * log(x[0]) - 1.26666 * log(x[1])
               - 0.0056822 * x[2] + 0.84874 * log(x[4]))
    shape = 8.555
    return Gamma(shape, shape / mean)
