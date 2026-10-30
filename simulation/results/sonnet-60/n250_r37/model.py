from adr.dist import Normal, exp


def model(x):
    t = x[0]
    m = ((-13.7037845 * t + 22.5939638) * t - 8.09819971) * t + 0.0177673651
    s = exp((4.76985564 * t - 3.2247201) * t - 0.786169906)
    return Normal(m, s)
