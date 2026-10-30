from adr.dist import Normal, Mixture, exp, maximum


def model(x):
    t = 2.0 * x[0] - 1.0
    h = maximum(x[0] - 0.3831, 0.0)
    base = ((-0.8238 * t + 2.0527) * t + 2.951) * t + 0.0602
    a = Normal(base + 0.5725 * h - 5.5326 * h * h, exp((-0.2338 * t - 0.7314) * t - 1.4585))
    b = Normal(base - 0.0963 * h - 13.6364 * h * h, exp((-0.2105 * t - 0.6895) * t - 1.5145))
    return Mixture([0.5471, 0.4529], [a, b])
