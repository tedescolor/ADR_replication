from adr.dist import Mixture, Normal, exp, sigmoid


def model(x):
    u = x[0] - 0.5
    lower = (((11.5266 * u - 15.1916) * u - 4.1928) * u + 2.6813) * u - 0.0889
    upper = 4.859 * u + 0.1843
    s_low = exp(-1.1743 * u - 1.5077)
    s_up = exp(-1.0281 * u - 1.5341)
    w = sigmoid(4.1548 * u - 1.1892)
    return Mixture([w, 1 - w], [Normal(upper, s_up), Normal(lower, s_low)])
