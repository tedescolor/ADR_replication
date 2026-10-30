from adr.dist import Mixture, Normal, exp, sigmoid


def model(x):
    t = x[0]
    w = sigmoid(-7.969+19.6706*t**1+-11.9957*t**2)
    m1 = -5.1832+22.0709*t**1+-35.4678*t**2+29.4814*t**3+-11.9012*t**4
    m2 = 0.0407+-7.9603*t**1+21.7174*t**2+-12.3479*t**3+1.0566*t**4
    s1 = exp(-0.4538+-1.9279*t**1)
    s2 = exp(-0.8818+-1.2272*t**1)
    return Mixture([w, 1 - w], [Normal(m1, s1), Normal(m2, s2)])
