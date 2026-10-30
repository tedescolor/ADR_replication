"""Two equally weighted Gaussian branches with polynomial means and a shared linear sd."""
from adr.dist import Mixture, Normal


def model(x):
    t = x[0]
    m1 = 0.081443 + t * (-10.216963 + t * (37.412346 + t * (-42.044517 + t * 13.836706)))
    m2 = 0.085505 + t * (-8.061862 + t * (20.927848 + t * -10.462886))
    s = 0.404117 - 0.310738 * t
    return Mixture([0.5, 0.5], [Normal(m1, s), Normal(m2, s)])
