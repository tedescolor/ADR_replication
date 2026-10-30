from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0] - 0.5
    w = sigmoid(((-13.97)*t+8.576)*t+-1.058)
    return Mixture(
        [1 - w, w],
        [Normal(((((14.48)*t+-15.81)*t+-5.11)*t+2.858)*t+-0.05439, exp(((-1.634)*t+-1.452)*t+-1.41)),
         Normal(((-2.143)*t+6.225)*t+-0.01511, exp((-2.101)*t+-1.241))])
