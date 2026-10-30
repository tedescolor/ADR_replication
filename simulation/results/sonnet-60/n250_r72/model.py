from adr.dist import Normal, exp


def model(x):
    t = x[0]
    m = -0.04495177 + t * (-7.25518808 + t * (20.8258068 + t * -12.89673042))
    s = exp(-0.65749492 + t * (-3.68161458 + t * 5.13775733))
    return Normal(m, s)
