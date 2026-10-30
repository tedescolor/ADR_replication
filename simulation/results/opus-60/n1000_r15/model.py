"""Two equally weighted normal branches: centre c(x) -/+ a gap opening at x0 = k.

c is a degree-6 polynomial in t = 2*x0 - 1, the half-gap is u*(b0*u + b1) with
u = max(0, x0 - k), and the shared log-sd is quadratic in t.
"""
from adr.dist import Normal, Mixture, exp, maximum


def model(x):
    t = 2.0 * x[0] - 1.0
    c = ((((((-0.794984195960259) * t + 0.20321697879232775) * t + 1.5750351403800407) * t + -1.748771926518247) * t + -0.418348262369333) * t + 1.9535891745412552) * t + -0.000974160582087522
    u = maximum(x[0] - 0.4750997864741322, 0.0)
    h = u * (3.827289699032061 * u + 1.3456410769300708)
    s = exp(((-0.35073747068813477) * t + -0.6036461153613499) * t + -1.4026238921269147)
    return Mixture([0.5, 0.5], [Normal(c - h, s), Normal(c + h, s)])
