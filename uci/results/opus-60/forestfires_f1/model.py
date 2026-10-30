"""Unconditional two-component mixture: a Normal lump near zero and a LogNormal right tail.
Covariates showed no out-of-sample predictive value, so the density does not depend on x."""
from adr.dist import Mixture, Normal, LogNormal


def model(x):
    return Mixture([0.444, 0.556], [Normal(-0.054, 0.607), LogNormal(1.572, 1.590)])
