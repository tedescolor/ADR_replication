from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    x0 = x[0]
    m1 = -0.20107 + x0*(-0.335329 + x0*(-26.795515 + x0*(104.82308 + x0*(-119.929 + x0*(45.025047)))))
    m2 = 0.160495 + x0*(-17.907263 + x0*(85.341544 + x0*(-150.483373 + x0*(118.985928 + x0*(-37.07748)))))
    s = exp(-1.303824 + x0*(0.667113 + x0*(-1.776336)))
    w = sigmoid(0.07174)
    return Mixture([w, 1 - w], [Normal(m1, s), Normal(m2, s)])
