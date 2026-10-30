from adr.dist import Normal, Mixture, sigmoid, exp


def model(x):
    x = x[0]
    p = sigmoid(((-3.0409)*x+5.7421)*x+-2.7376)
    return Mixture(
        [1 - p, p],
        [Normal((((((-4.7677)*x+9.1874)*x+-15.346)*x+20.884)*x+-7.5607)*x+0.035399, exp(((-1.1255)*x+-0.25288)*x+-1.0323)),
         Normal((((-8.5005)*x+9.1917)*x+-0.77291)*x+-0.91969, exp((-1.0552)*x+-1.0888))])
