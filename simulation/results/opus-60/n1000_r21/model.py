from adr.dist import Normal, Mixture, exp


def model(x):
    t = x[0] - 0.5
    c = -0.01451 + t * (3.69764 + t * (-0.36478 + t * (-10.66917 + t * (9.09894 + t * (-6.72869)))))
    h = -0.02181 + t * (1.33733 + t * (6.22896 + t * (-2.69827 + t * (-9.60673 + t * (16.04425)))))
    s = exp(-1.46679 + t * (-1.26478 + t * (-1.7683)))
    return Mixture([0.5, 0.5], [Normal(c + h, s), Normal(c - h, s)])
