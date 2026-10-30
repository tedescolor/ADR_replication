from adr.dist import Normal, exp


def model(x):
    t = x[0]
    m = 0.26066936 + t * (-13.14485723 + t * (47.08015697 + t * (-54.78076854 + t * 21.8465869)))
    s = exp(-0.55441118 + t * (-3.86171114 + t * 5.15390932))
    return Normal(m, s)
