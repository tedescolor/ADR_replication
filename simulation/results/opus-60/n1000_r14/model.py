from adr.dist import Normal, StudentT, Mixture, exp, log, sigmoid

def model(x):
    t = x[0]
    mA = (((((12.438548)*t + -32.881078)*t + 21.797904)*t + 6.414443)*t + -5.113457)*t + -0.119073
    h = log(1 + exp((t - 0.518801) * 403.428793)) / 403.428793
    mB = mA - (((8.419385)*h + 1.048974)*h + 4.931128) * h
    sA = exp(((-1.615035)*t + 0.591997)*t + -1.357875)
    sB = exp(((-0.656948)*t + -0.944145)*t + -0.67521)
    w = sigmoid(0.015882)
    return Mixture([w, 1 - w], [Normal(mA, sA), Normal(mB, sB)])
