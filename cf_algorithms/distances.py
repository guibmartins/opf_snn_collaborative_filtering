import numpy as np
from numba import njit

EPSILON = 1e-20


@njit(cache=True)
def cosine(x, y):

    norm_factor = (np.sum(x ** 2) ** 0.5) * (np.sum(y ** 2) ** 0.5)
    return 1 - (np.sum(x * y) / (norm_factor + EPSILON))


@njit(cache=True)
def euclidean(x, y):

    dist = (x - y) ** 2
    return np.sum(dist) ** 0.5


@njit(cache=True)
def pearson(x, y):

    mu_x = np.mean(x)
    mu_y = np.mean(y)

    norm_factor = np.sum((x - mu_x) ** 2) ** 0.5 * (np.sum((y - mu_y) ** 2) ** 0.5)
    return 1 - (np.sum((x - mu_x) * (y - mu_y)) / (norm_factor + EPSILON))


@njit(cache=True)
def squared_euclidean(x, y):

    dist = (x - y) ** 2
    return np.sum(dist)


# @njit(cache=True)
def significance_weighting(function, xi, xj, beta=1):

    w = function(xi, xj)

    return w * (np.minimum(len((xi > 0) & (xj > 0)), beta) / beta)


# @njit(cache=True)
def amplified_weighting(function, xi, xj, alpha=1):
    w = function(xi, xj)

    return w ** alpha


# A distances constant dictionary for selecting the desired
# distance metric to be used
DISTANCES = {
    'cosine': cosine,
    'euclidean': euclidean,
    'pearson': pearson,
    'squared_euclidean': squared_euclidean,
}