from adr.dist import Normal, exp


def model(x):
    t = x[0]
    m = -0.12532397 + t * (-6.72287094 + t * (19.87222596 + t * -12.44213687))
    s = exp(-0.7536893 + t * (-3.32010673 + t * 4.8376004))
    return Normal(m, s)
