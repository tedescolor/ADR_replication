from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    p = sigmoid(3.4075+t*(-7.2081+t*3.5564))
    a = Normal(-0.0037731+t*(-7.9698+t*(21.1+t*-10.619)), exp(-0.91777+-1.1595*t))
    b = Normal(-0.22582+t*(-3.9919+t*(13.979+t*-10.757)), exp(-0.85445+-1.3283*t))
    return Mixture([p, 1 - p], [a, b])
