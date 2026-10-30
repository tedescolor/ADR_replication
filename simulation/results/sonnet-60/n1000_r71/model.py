from adr.dist import Mixture, Normal, exp, sigmoid


def model(x):
    x = x[0]
    w = sigmoid(((-11.787)*x+(19.671))*x+(-8.2451))
    return Mixture(
        [1 - w, w],
        [Normal(((((-2.2098)*x+(-4.5355))*x+(15.365))*x+(-5.9889))*x+(-0.12872), exp((-1.3165)*x+(-0.79905))),
         Normal(((((-1.8649)*x+(-2.1783))*x+(0.53515))*x+(4.8121))*x+(-2.2932), exp((-1.8944)*x+(-0.5106)))],
    )
