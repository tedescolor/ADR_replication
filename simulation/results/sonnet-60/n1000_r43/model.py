from adr.dist import Normal, exp


def model(x):
    x0 = x[0]
    mu = -0.0575998*x0**0+-3.33504*x0**1+-30.5804*x0**2+218.676*x0**3+-463.866*x0**4+424.003*x0**5+-144.326*x0**6
    ls = -0.682059*x0**0+-3.66124*x0**1+5.1762*x0**2
    return Normal(mu, exp(ls))
