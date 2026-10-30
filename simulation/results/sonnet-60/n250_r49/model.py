from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    w = sigmoid(-3.4174 + 4.2396 * t)
    ma = -0.1683 + t * (-6.3139 + t * (20.9598 + t * -15.6515))
    mb = -2.6084 + 5.2823 * t
    return Mixture([1 - w, w], [Normal(ma, 0.2731), Normal(mb, 0.1244)])
