from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    z = ((-15.2785)*t+25.2308)*t+-9.963
    return Mixture(
        [1 - sigmoid(z), sigmoid(z)],
        [Normal(((((-1.0974)*t+-7.5679)*t+18.3)*t+-7.1588)*t+0.0233, exp(((-0.974)*t+-0.252)*t+-1.0157)),
         Normal(((-10.8541)*t+14.6568)*t+-4.803, exp((-1.8883)*t+-0.4792))],
    )
