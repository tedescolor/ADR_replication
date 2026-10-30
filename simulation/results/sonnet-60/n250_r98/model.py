from adr.dist import Normal, exp


def model(x):
    u = x[0]
    mu = 0.03310624 + u * (-8.18564137 + u * (23.20670436 - 14.61371948 * u))
    s = exp(-0.6870419 + u * (-3.06452536 + 4.43503869 * u))
    return Normal(mu, s)
