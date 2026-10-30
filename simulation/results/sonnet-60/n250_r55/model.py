from adr.dist import Normal, exp


def model(x):
    t = x[0]
    m = -0.05805103 + t * (-7.43089602 + t * (21.61156177 + t * -13.60324988))
    s = exp(-0.85610908 + t * (-2.81086649 + t * 4.40535822))
    return Normal(m, s)
