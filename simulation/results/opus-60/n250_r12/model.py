from adr.dist import Normal, Mixture, exp, maximum


def model(x):
    """Two-branch mixture: shared cubic trend; branch b forks off below it after x ~ 0.495."""
    x = x[0]
    base = -0.058791 - 7.489639 * x + 20.302318 * x ** 2 - 10.205316 * x ** 3
    h = maximum(x - 0.494839, 0.0)
    w = 0.493495
    a = Normal(base, exp(-0.88756 - 1.284499 * x))
    b = Normal(base - 3.774023 * h - 6.344462 * h * h, exp(-0.829748 - 1.362669 * x))
    return Mixture([w, 1 - w], [a, b])
