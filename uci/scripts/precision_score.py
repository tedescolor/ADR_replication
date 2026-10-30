"""The score of a program: the precision-aware negative log-likelihood.

The target is recorded at a finite precision w. A program is scored on the probability it gives
to an interval of width w around each held-out value y:

    score(y) = -log P(y - w/2 < Y <= y + w/2 | x) + log w,     Y ~ model(x).

For a density that is smooth at scale w this equals the usual negative log-likelihood
-log p(y | x). Structure finer than w (for example, components narrower than the recording
precision) earns nothing: the best attainable value per row is log w.

    from precision_score import precision_nll
    precision_nll(model, X, y, w)        # mean score over the rows of X, y
"""
import numpy as np

from adr import dist


def bin_mass(d, a, b):
    """P(a < Y <= b) for a univariate Distribution, accurate in both tails: mixtures are split
    into their components, and each component uses its survival function in its upper half,
    which avoids the cancellation 1 - 1 far in the right tail."""
    if isinstance(d, dist.Mixture):
        return sum(float(np.ravel(d.w[k])[0]) * bin_mass(c, a, b) for k, c in enumerate(d.components))
    f = d.d  # frozen scipy distribution
    if float(np.ravel(f.cdf(a))[0]) < 0.5:
        return float(np.ravel(f.cdf(b))[0] - np.ravel(f.cdf(a))[0])
    return float(np.ravel(f.sf(a))[0] - np.ravel(f.sf(b))[0])


def row_score(d, y, w):
    """Score of one distribution d at one recorded value y."""
    mass = bin_mass(d, y - w / 2, y + w / 2)
    if not mass > 1e-250:  # both tails underflow: the density is flat at scale w there
        return -float(np.ravel(d.logpdf(np.array([y])))[0])
    return -np.log(mass) + np.log(w)


def precision_nll(model, X, y, w):
    """Mean precision-aware negative log-likelihood of `model` on rows (X, y)."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    return float(np.mean([row_score(model(x), yi, w) for x, yi in zip(X, y)]))
