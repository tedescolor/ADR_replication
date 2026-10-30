from adr.dist import Normal, exp


def model(x):
    t = x[0]
    mu = ((-13.68726358 * t + 21.48970592) * t - 7.16656421) * t - 0.13828213
    s = exp((4.64052022 * t - 2.94005981) * t - 0.92340606)
    return Normal(mu, s)
