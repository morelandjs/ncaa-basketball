#!/usr/bin/env python3

from pathlib import Path

import numpy as np
from skopt import gp_minimize

from melo import Melo
from ncaabb_games import games
from xdg import XDG_DATA_HOME


cachedir = Path(XDG_DATA_HOME, 'nba')
cachedir.mkdir(parents=True, exist_ok=True)
cachefile = cachedir / 'games.pkl'

dates = games['date']
labels1 = games['home_team']
labels2 = games['away_team']
spreads = games['home_points'] - games['away_points']
totals = games['home_points'] + games['away_points']


def melo_wrapper(mode, k, bias, smooth, regress, verbose=False):
    """
    Wrapper to pass arguments to the Melo library.

    """
    values, commutes, lines = {
        'spread': (spreads, False, np.arange(-60.5, 61.5)),
        'total': (totals, True, np.arange(-115.5, 300.5)),
    }[mode]

    biases = bias * np.logical_not(games['neutral'])

    return Melo(
        dates, labels1, labels2, values, lines=lines,
        k=k, biases=biases, smooth=smooth, commutes=commutes,
        regress=lambda t: regress if t > np.timedelta64(12, 'W') else 0
    )


def from_cache(mode, retrain=False, **kwargs):
    """
    Load the melo args from the cache if available, otherwise
    train and cache a new instance.

    """
    cachefile = cachedir / '{}.cache'.format(mode.lower())

    if not retrain and cachefile.exists():
        args = np.loadtxt(cachefile)
        return melo_wrapper(mode, *args)

    def obj(args):
        melo = melo_wrapper(mode, *args)
        return melo.entropy()

    bounds = {
        'spread': [
            (0.0,    0.5),
            (0.0,    0.5),
            (0.0,   15.0),
            (0.0,    0.5),
        ],
        'total': [
            (0.0,    0.5),
            (-0.01, 0.01),
            (0.0,   15.0),
            (0.0,    0.5),
        ],
    }[mode]

    res = gp_minimize(obj, bounds, n_calls=40, n_jobs=4, verbose=True)

    print("mode: {}".format(mode))
    print("best mean absolute error: {:.4f}".format(res.fun))
    print("best parameters: {}".format(res.x))

    if not cachefile.parent.exists():
        cachefile.parent.mkdir()

    np.savetxt(cachefile, res.x)
    return melo_wrapper(mode, *res.x)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='calibrate model parameters for point spreads and totals',
        argument_default=argparse.SUPPRESS
    )

    parser.add_argument(
        '--retrain', action='store_true', default=False,
        help='retrain even if model args are cached'
    )

    args = parser.parse_args()
    kwargs = vars(args)

    for mode in 'spread', 'total':
        from_cache(mode, **kwargs)
else:
    ncaabb_spreads = from_cache('spread')
    ncaabb_totals = from_cache('total')
