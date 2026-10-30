from adr.dist import Normal, Mixture, exp, maximum, sigmoid


def model(x):
    t = x[0]
    m1 = -0.014091 + t * (-7.966299 + t * (23.92607 + t * (-21.300933 + t * (13.37721 + t * (-5.526453)))))
    h = maximum(t - 0.479085, 0.0)
    d = -exp(2.234885) * h ** exp(0.413938)
    s1 = exp(-1.055384 + t * (-0.186665 + t * (-1.041753)))
    s2 = exp(-1.049906 + t * (-0.049258 + t * (-1.216334)))
    w = sigmoid(0.064842)
    return Mixture([1.0 - w, w], [Normal(m1, s1), Normal(m1 + d, s2)])
