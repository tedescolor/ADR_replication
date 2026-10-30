"""Two-component Normal mixture: one branch for x < ~0.53, a second (rising) branch switching on beyond."""
from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    z = x[0] - 0.5
    pw = sigmoid(958.1955147967727 * (z - 0.028528018612977133)) * sigmoid(-0.08034394025859741)
    mu_a = (((((((181.77583826632008) * z - 112.58183227100227) * z - 58.577595807718566) * z + 46.58784479934458) * z - 10.235314889898417) * z - 6.872085906520664) * z + 2.605032597829659) * z - 0.05743241962868857
    sd_a = exp(((-1.5049084011668028) * z - 1.5303642619749052) * z - 1.358679243181173)
    mu_b = ((-4.667638471414194) * z + 7.785665120582112) * z - 0.20296126060923284
    sd_b = exp(((4.1724423331441844) * z - 5.234627017343634) * z - 0.8498682056512027)
    return Mixture([1 - pw, pw], [Normal(mu_a, sd_a), Normal(mu_b, sd_b)])
