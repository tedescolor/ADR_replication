from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0] - 0.5
    w = sigmoid((-0.83113)*t + 0.386124)
    a = Normal(((((2.508117)*t + -10.036134)*t + 4.049299)*t + 5.240835)*t + -0.020265, exp(((-2.126755)*t + -1.117244)*t + -1.414292))
    b = Normal(((((7.754882)*t + -15.45488)*t + -2.180178)*t + 2.286146)*t + -0.128218, exp(((-1.648846)*t + -0.804267)*t + -1.495065))
    return Mixture([w, 1 - w], [a, b])
