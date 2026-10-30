from adr.dist import Normal, Mixture, sigmoid, exp


def model(x):
    x = x[0]
    p = sigmoid(-3.484 + 7.758*x - 4.241*x**2)
    return Mixture(
        [1 - p, p],
        [Normal(0.101 - 8.304*x + 21.64*x**2 - 10.93*x**3, exp(-0.9714 - 0.1561*x - 1.225*x**2)),
         Normal(-0.9326 - 0.4105*x + 8.498*x**2 - 8.118*x**3, exp(-2.2 + 2.66*x - 2.892*x**2))])
