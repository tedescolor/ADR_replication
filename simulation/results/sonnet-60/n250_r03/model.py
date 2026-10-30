from adr.dist import Normal, exp


def model(x):
    t = x[0]
    m = ((-15.25337164 * t + 23.8223385) * t - 8.19568419) * t + 0.04291007
    s = exp((5.92679851 * t - 4.54916901) * t - 0.46861585)
    return Normal(m, s)
