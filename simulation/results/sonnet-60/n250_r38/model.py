from adr.dist import Normal, exp


def model(x):
    t = x[0]
    m = ((-14.70163659 * t + 22.57215519) * t - 7.41205996) * t - 0.11464572
    s = exp((5.07400479 * t - 3.51323853) * t - 0.73259945)
    return Normal(m, s)
