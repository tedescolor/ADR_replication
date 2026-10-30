from adr.dist import Normal, exp


def model(x):
    t = x[0]
    m = 0.19167 + t * (-10.71275 + t * (37.06166 + t * (-41.01689 + t * 15.5263)))
    s = exp(-0.68792 + t * (-3.48571 + t * 4.96609))
    return Normal(m, s)
