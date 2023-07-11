"""Distance-based metrics.
"""

import math

import numpy as np
from numba import njit

import opfython.utils.constants as c
import opfython.utils.decorator as d


@d.avoid_null_features
@njit(cache=True)
def arc_cosine(x, y):

    if not len(x):
        return 1.0

    norm_factor = (np.sum(x ** 2) ** 0.5) * (np.sum(y ** 2) ** 0.5)

    return np.arccos(np.sum(x * y) / (norm_factor + c.EPSILON)) / np.pi


@d.avoid_null_features
@njit(cache=True)
def cosine(x, y):

    if not len(x):
        return 2.

    norm_factor = (np.sum(x ** 2) ** 0.5) * (np.sum(y ** 2) ** 0.5)
    return 1 - (np.sum(x * y) / (norm_factor + c.EPSILON))


@d.avoid_null_features
@njit(cache=True)
def euclidean(x, y):

    if not len(x):
        return 1.0 #c.MAX_ARC_WEIGHT

    dist = (x - y) ** 2

    return (np.sum(dist) ** 0.5) / float(len(x))


@njit(cache=True)
def jaccard(x, y):

    inter = np.sum((x > 0) & (y > 0)) + 0.
    union = np.sum((x > 0) | (y > 0)) + 0.

    return 1. - (inter / (float(union) + c.EPSILON))


@d.avoid_null_features
@njit(cache=True)
def log_squared_euclidean(x, y):

    if not len(x):
        return c.MAX_ARC_WEIGHT

    dist = np.sum((x - y) ** 2) / float(len(x))

    return c.MAX_ARC_WEIGHT * math.log(dist + 1)


@njit(cache=True)
def pearson(x, y):

    idx = (x > 0) & (y > 0)

    if not len(idx):
        return 2.

    mu_x = np.mean(x[x > 0])
    mu_y = np.mean(y[y > 0])

    sim = np.sum((x[idx] - mu_x) * (y[idx] - mu_y)) + 0.
    norm_factor = (np.sum((x[idx] - mu_x) ** 2) ** 0.5) * (np.sum((y[idx] - mu_y) ** 2) ** 0.5) + c.EPSILON

    return 1. - (sim / norm_factor)


@d.avoid_null_features
@njit(cache=True)
def squared_euclidean(x, y):

    if not len(x):
        return c.MAX_ARC_WEIGHT

    dist = (x - y) ** 2

    return np.sum(dist) / float(len(x))


# A distances constant dictionary for selecting
# the desired distance metric to be used
DISTANCES = {
    'arc_cosine': arc_cosine,
    'cosine': cosine,
    'euclidean': euclidean,
    'jaccard': jaccard,
    'log_squared_euclidean': log_squared_euclidean,
    'pearson': pearson,
    'squared_euclidean': squared_euclidean,
}
