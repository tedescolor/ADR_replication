from adr.dist import Mixture, Normal, exp, sigmoid


def model(x):
    u = x[0] - 0.5
    ma = -0.045684 + 4.8383*u**1 + 6.2184*u**2 + -9.407*u**3 + -4.3523*u**4
    mb = -0.084922 + 2.8185*u**1 + -8.9527*u**2
    sa = exp(-1.4051 + -1.3671*u**1 + -1.2895*u**2)
    sb = exp(-1.3772 + -1.7056*u**1)
    p = sigmoid(-1.4812 + 4.8045*u)
    return Mixture([1 - p, p], [Normal(ma, sa), Normal(mb, sb)])
