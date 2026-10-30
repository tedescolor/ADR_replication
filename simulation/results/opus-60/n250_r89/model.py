from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    wa = sigmoid((-2.450246) * t + 1.734743)
    return Mixture([wa, 1.0 - wa], [
        Normal((((-12.466701) * t + 23.86806) * t + -9.248761) * t + 0.211341, exp((-1.725552) * t + -0.762814)),
        Normal((((-9.237281) * t + 10.942504) * t + -2.247815) * t + -0.468326, exp((-1.552141) * t + -0.693453)),
    ])

