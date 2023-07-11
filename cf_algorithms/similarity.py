import utils.constants as c
import numpy as np
from numba import njit


# @njit(cache=True)
def _intersection(x, y):
    return (x > 0) & (y > 0)


def _union(x, y):
    return (x > 0) | (y > 0)


# @njit(cache=True)
def cosine(x, y):
    indices = _intersection(x, y)

    if not np.sum(indices):
        return c.MIN_COSINE

    sim = np.sum(x[indices] * y[indices]) + 0
    norm_factor = (np.sum(x[indices] ** 2) ** 0.5) * (np.sum(y[indices] ** 2) ** 0.5) + c.EPSILON

    return sim / norm_factor


@njit(cache=True)
def pearson(x, y):

    indices = (x > 0) & (y > 0)

    if not len(indices):
        return c.MIN_PEARSON

    mu_x = np.mean(x[x > 0])
    mu_y = np.mean(y[y > 0])

    sim = np.sum((x[indices] - mu_x) * (y[indices] - mu_y)) + 0.
    norm_factor = (np.sum((x[indices] - mu_x) ** 2) ** 0.5) * (np.sum((y[indices] - mu_y) ** 2) ** 0.5)

    return sim / (norm_factor + c.EPSILON)


@njit(cache=True)
def jaccard(x, y):

    inter = np.sum((x > 0) & (y > 0)) + 0.
    union = np.sum((x > 0) | (y > 0)) + 0.

    return inter / (float(union) + c.EPSILON)


def shared_near_neighbor_distance(x, y):

    sim = (len(set(x) & set(y)) + 2.) / \
          float(len(set(x) | set(y)) + 2.)

    return sim


# @njit(cache=True)
def weighted_voting_shared_neighbors(x, y):

    if len(x) < len(y):
        k = len(x) - 1
    else:
        k = len(y) - 1

    w = 0.

    shared_neighbors = list(set(x) & set(y))

    for i in shared_neighbors[:k]:

        r = np.argwhere(np.array(x) == i)[0, 0]
        q = np.argwhere(np.array(y) == i)[0, 0]
        w += (k - r + 1) * (k - q + 1)

    # return 1 / float(w)
    # w_norm = np.sum([(k - i + 1) ** 2 for i in range(len(x[:k]))])
    # return 1 - (w / float(w_norm))
    return float(w)


SIMILARITIES = {
    'cosine': cosine,
    'jaccard': jaccard,
    'pearson': pearson,
    'snn': shared_near_neighbor_distance,
    'wv_snn': weighted_voting_shared_neighbors
}

SIMILARITY_INTERVAL = {
    'cosine': (c.MIN_COSINE, c.MAX_COSINE),
    'jaccard': (c.MIN_JACCARD, c.MAX_JACCARD),
    'pearson': (c.MIN_PEARSON, c.MAX_PEARSON)
}
