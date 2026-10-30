from adr.dist import Normal, Mixture, maximum, abs


def model(x):
    """Two equally weighted Gaussian branches with a shared noise sd that is linear in x.
    Both follow the cubic mu below the split point x0; above it the lower branch drops
    below mu by a quadratic gap in (x - x0)."""
    t = 2 * x[0] - 1
    mu = ((-1.27941 * t + 1.26296) * t + 2.5169) * t + -0.00152
    h = maximum(x[0] - 0.52656, 0)
    gap = (4.6105 + 5.83201 * h) * h
    s = abs(0.39017 + -0.3036 * x[0]) + 0.001
    return Mixture([0.5, 0.5], [Normal(mu, s), Normal(mu - gap, s)])
