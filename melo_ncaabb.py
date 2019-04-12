#!/usr/bin/env python3

from melo import Melo
from ncaabb_games import games as g
import numpy as np
from pyDOE import lhs


def design(bounds, samples):
    """
    Latin hypercube experiment design

    """
    xmin, xmax = map(np.array, zip(*bounds))
    ndim = len(bounds)

    return xmin + (xmax - xmin) * lhs(ndim, samples=samples)


def melo_wrapper(k, bias, smooth, regress):
    """
    Wrapper to pass arguments to the Melo library.

    """
    print(k, bias, smooth, regress)

    bias *= np.logical_not(g.neutral)

    return Melo(
        g.date, g.home_team, g.away_team, g.home_points - g.away_points,
        lines=np.arange(-70.5, 71.5), k=k, bias=bias, smooth=smooth,
        regress=lambda t: regress*(t > 3), regress_unit='month'
    )


def optimize(bounds, samples=50):
    """
    Estimate optimal model parameters using cross entropy

    """
    X = design(bounds, samples=samples)
    y = [melo_wrapper(*x).entropy for x in X]

    return X[np.argmin(y)]


if __name__ == "__main__":
    bounds = [(0, 0.5), (0, 0.5), (0, 15), (0, 0.5)]
    args = optimize(bounds, samples=1000)
    print(args)
else:
    ncaabb_spreads = melo_wrapper(.286, .38, 4.0, 0.03)
