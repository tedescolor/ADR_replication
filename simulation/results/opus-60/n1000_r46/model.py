from adr.dist import Normal, Mixture, abs


def model(x):
    """Equal-weight two-branch Gaussian mixture; branch means are degree-6 polynomials
    in t = 2*x0 - 1, common noise sd linear in t."""
    t = 2.0 * x[0] - 1.0
    m1 = ((((((-0.8414690678203613) * t + 0.1457077496869752) * t + 0.7796970158623683) * t + -1.112958918803902) * t + 1.1312856310513504) * t + 2.264743483797501) * t + 0.06918079163233778
    m2 = ((((((-2.052470358204592) * t + 0.25983790518657235) * t + 4.04878035963989) * t + -2.550070697371669) * t + -2.361033108216185) * t + 1.7057199499724272) * t + -0.09284003910096661
    s = abs((-0.13188975741060513) * t + 0.21903811274442506)
    return Mixture([0.5, 0.5], [Normal(m1, s), Normal(m2, s)])
