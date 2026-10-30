from adr.dist import Normal, Mixture, exp, log


def model(x):
    """Two equal-weight Normal branches with a shared sd; the upper branch separates
    from the lower one through a smooth softplus gap (zero before the split)."""
    u = 2 * x[0] - 1
    m1 = ((((((-1.246829) * u + 0.3184571) * u + 2.389516) * u + -2.276128) * u + -1.564396) * u + 1.431212) * u + -0.08863083
    g = log(1 + exp(54.69602 * (u - -0.06003999))) / 54.69602
    m2 = m1 + (1.933157 * g + 1.273329) * g
    sd = exp(((-0.2705538) * u + -0.6828607) * u + -1.399558)
    return Mixture([0.5, 0.5], [Normal(m1, sd), Normal(m2, sd)])
