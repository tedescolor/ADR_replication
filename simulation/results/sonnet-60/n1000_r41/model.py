from adr.dist import Mixture, Normal, exp


def model(x):
    """Two-branch Normal mixture; polynomial means, log-scales, logistic weight."""
    x = x[0]
    z = -4.2974 + 9.7049*x + -5.3378*x*x
    p = 1 / (1 + exp(-z))
    return Mixture(
        [p, 1 - p],
        [
            Normal(-1.6688 + 3.4883*x + 0.47778*x*x + -0.84928*x*x*x + -2.4573*x*x*x*x, exp(-4.9397 + 9.4654*x + -7.066*x*x)),
            Normal(-0.10305 + -6.2004*x + 15.638*x*x + -4.4336*x*x*x + -2.4072*x*x*x*x, exp(-1.0092 + -0.072941*x + -1.2777*x*x)),
        ],
    )
