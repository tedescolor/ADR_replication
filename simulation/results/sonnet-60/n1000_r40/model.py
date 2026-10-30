from adr.dist import Mixture, Normal, exp, sigmoid


def model(x):
    x0 = x[0]
    w = sigmoid(-17.064 + 58.89*x0 + -66.847*x0**2 + 24.911*x0**3)
    m1 = -7.8726 + 31.285*x0 + -45.017*x0**2 + 31.717*x0**3 + -11.103*x0**4
    m2 = 0.03588 + -8.1901*x0 + 22.828*x0**2 + -13.716*x0**3 + 1.5317*x0**4
    s1 = exp(-1.1117 + -1.0505*x0)
    s2 = exp(-0.84751 + -1.4397*x0)
    return Mixture([w, 1 - w], [Normal(m1, s1), Normal(m2, s2)])
