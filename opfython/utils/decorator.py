"""Decorators.
"""

from functools import wraps

import opfython.utils.constants as c


def avoid_zero_division(f):
    """Adds a minimal value to arguments to avoid zero values.

    Args:
        f (callable): Incoming function.

    Returns:
        The incoming function with its adjusted arguments.

    """

    @wraps(f)
    def _avoid_zero_division(x, y):
        """Wraps the function for adjusting its arguments

        Returns:
            The function itself.

        """

        x += c.EPSILON
        y += c.EPSILON

        return f(x, y)

    return _avoid_zero_division


def avoid_null_features(f):
    @wraps(f)
    def _avoid_null_features(x, y):
        idx = (x > 0) & (y > 0)
        return f(x[idx], y[idx])

    return _avoid_null_features
