from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    u = x[0] - 0.5
    w = sigmoid(-1.6226 + 4.2461 * u)
    a = Normal(((((13.984)*u+-14.878)*u+-5.009)*u+2.6665)*u+-0.044318, exp(-1.5016 + -1.2333 * u))
    b = Normal(((-1.5705)*u+5.8228)*u+0.041434, exp(-1.4097 + -1.4249 * u))
    return Mixture([1 - w, w], [a, b])
