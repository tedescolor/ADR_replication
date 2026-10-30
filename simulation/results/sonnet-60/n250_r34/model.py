from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    ma = -0.13435 + t * (-5.52903 + t * (1.90363 + t * (57.2746 + t * (-99.1711 + t * 44.7663))))
    sa = exp(-0.959227 - 0.98107 * t)
    mb = -3.41916 + t * (7.87107 - 1.86748 * t)
    pb = sigmoid(-3.75529 + 4.11826 * t)
    return Mixture([1 - pb, pb], [Normal(ma, sa), Normal(mb, exp(-1.99482))])
