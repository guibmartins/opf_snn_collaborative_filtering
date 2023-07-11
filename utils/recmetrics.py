import numpy as np


# Discount Cumulative Gain (DCG@k)
def _dcg_score(rank, k=None, form='trad'):
    """
        Discounted Cumulative Gain at rank k

        Args:
                rank: Relevance scores in decreasing rank order
                k: Number of items to be considered
                form: equation to be used

                        form='trad' : traditional formulation

                            dcg = sum(rel_i / log2(i + 1)),
                                where i = 1 to k; or
                            dcg = rel_1 + sum(rel_i / log2(i)),
                                where i = 2 to k

                        form='alt' : alternative formulation

                            dcg = sum( (2^rel_i) - 1 / log2(i + 1) ),
                                where i = 1 to k

        Returns:

                Discounted cumulative gain score for top k relevant items
        """
    if not isinstance(k, (int, np.int32, np.int64)):
        k = 1

    if not isinstance(form, str):
        form = 'trad'

    # Select only the top-k ranked scores
    rank_at_k = np.asfarray(rank)[:k]

    # Calculate the accumulated discount for all scores (relevance)
    rank_weight = 1. / np.log2(np.arange(2, rank_at_k.size + 2))

    if rank_at_k.size > 0:

        if form == 'trad':
            return np.sum(rank_weight * rank_at_k, axis=-1)
        elif form == 'alt':
            return np.sum((2 ** rank_at_k - 1.) * rank_weight, axis=-1)
        else:
            raise ValueError('Formulation parameter must be `trad` or `alt`.')

    return 0.


# Normalized Discounted Cumulative Gain (NDCG@k)
def ndcg_score(rank_true, rank_pred, k=None, form='trad'):
    """
    Normalized Discounted Cumulative Gain at rank k

    Args:
            rank_true: Ground truth relevance scores (decreasing rank order)
            rank_pred: Predicted relevance scores (decreasing rank order)
            k: Number of items to be considered
            form: equation to be used
                form='trad' : traditional formulation

                    dcg = sum(rel_i / log2(i + 1)),
                        where i = 1 to k; or
                    dcg = rel_1 + sum(rel_i / log2(i)),
                        where i = 2 to k

                form='alt' : alternative formulation

                    dcg = sum( (2^rel_i) - 1 / log2(i + 1) ),
                        where i = 1 to k

            ndcg = dcg_score/ ideal_dcg
                where ndcg within [0-1]

    Returns:

            Normalized Discounted cumulative gain score for top k scores
    """

    if not isinstance(k, (int, np.int32, np.int64)):
        k = 1

    if not isinstance(form, str):
        form = 'trad'

    assert rank_true.size == rank_pred.size
    assert k <= rank_true.size

    if rank_pred.size > 0:
        # Calculate the ideal DCG, i.e. the ground truth ranked order
        idcg = _dcg_score(rank_true, k=k, form=form)

        # Calculate the predicted scores DCG
        dcg = _dcg_score(rank_pred, k=k, form=form)
        return dcg / idcg if idcg > 0. else 0.

    return 0.


# # Old implementation
# def dcg_at_k(rank, k=1, method=0):
#     """
#     Discounted Cumulative Gain at rank k
#
#     Args:
#
#             rank: Relevance scores in decreasing rank order
#             k: Number of items to be considered
#             method: equation to be used
#
#                     method=0 : traditional formulation
#
#                         dcg = sum(rel_i / log2(i + 1)),
#                             where i = 1 to k; or
#                         dcg = rel_1 + sum(rel_i / log2(i)),
#                             where i = 2 to k
#
#                     method=1 : alternative formulation
#
#                         dcg = sum( (2^rel_i) - 1 / log2(i + 1) ),
#                             where i = 1 to k
#
#     Returns:
#
#             Discounted cumulative gain score for top k relevant items
#     """
#
#     assert k <= rank.size
#     if k is None: k = rank.size
#
#     # filtering the top k scores
#     rank_at_k = np.asfarray(rank)[:k]
#     rank_weight = 1. / np.log2(np.arange(2, rank_at_k.size + 2))
#
#     if rank_at_k.size > 0:
#         if method == 0:
#             return np.sum(rank_at_k * rank_weight, axis=-1)
#             # return rank_at_k[0] + np.sum(rank_at_k[1:] / np.log2(np.arange(2, rank_at_k[1:].size + 2)))
#         elif method == 1:
#             return np.sum((2 ** rank_at_k - 1.) * rank_weight, axis=-1)
#             # return np.sum(((2 ** rank_at_k) - 1.) / np.log2(np.arange(2, rank_at_k.size + 2)))
#         else:
#             raise ValueError('Method must be 0 or 1.')
#
#     return 0.
#
#
# # Old implementation
# def ndcg_at_k(rank_true, rank_pred, k=None, method=0):
#     """
#     Normalized Discounted Cumulative Gain at rank k
#
#     Args:
#
#             rank_true: Ground truth relevance scores (decreasing rank order)
#             rank_pred: Predicted relevance scores (decreasing rank order)
#             k: Number of items to be considered
#
#             ndcg = dcg_score/ ideal_dcg
#                 where ndcg within [0-1]
#
#
#     Returns:
#
#             Normalized Discounted cumulative gain score for top k scores
#     """
#
#     assert len(rank_true) == len(rank_pred)
#     assert k <= rank_true.size
#     if k is None: k = rank_true.size
#
#     if rank_pred.size > 0:
#         # idcg = dcg_at_k(np.array(sorted(rank_true, reverse=True)), k=k, method=1)
#         idcg = dcg_at_k(rank_true, k=k, method=method)
#         return dcg_at_k(rank_pred, k=k, method=method) / idcg if idcg > 0. else 0.
#
#     return 0.


def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def r_precision(rank):
    """
    Score is precision after all relevant documents have been retrieved
    Relevance is binary (nonzero is relevant).
    >>> rank = [0, 0, 1]
    >>> r_precision(rank)
    0.33333333333333331
    >>> rank = [0, 1, 0]
    >>> r_precision(rank)
    0.5
    >>> rank = [1, 0, 0]
    >>> r_precision(rank)
    1.0

    Args:
        rank: Relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        R Precision
    """
    rank = np.asarray(rank) != 0
    z = rank.nonzero()[0]

    if z.size > 0.:
        return np.mean(rank[:z[-1] + 1])

    return 0.


def precision_recall_at_k(y_true, y_pred, k=1, relevance=1.):
    """
    Calculate Precision for a list of recommendation at top-k positions

    Args:
        y_true (np.array):
        y_pred (np.array):
        k (int):
        relevance (float):

    Return:
        precision (float)
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if not isinstance(k, (int, np.int32, np.int64)):
        k = 1

    if k > len(y_true):
        ValueError('`k` must be equal or small than the size of `y_true`.')

    if not isinstance(relevance, (float, np.float32, np.float64)):
        relevance = 1.

    arg_true_rank = np.argsort(y_true)[::-1][:k]
    arg_pred_rank = np.argsort(y_pred)[::-1][:k]

    relevant = arg_true_rank[np.argwhere(y_true[arg_true_rank] >= relevance).T[0]]
    recommended = arg_pred_rank[np.argwhere(y_pred[arg_pred_rank] >= relevance).T[0]]
    recommended_relevant = np.intersect1d(recommended, relevant)

    precision = len(recommended_relevant) / float(len(recommended)) if len(recommended) > 0 else 1.
    recall = len(recommended_relevant) / float(len(relevant)) if len(relevant) > 0 else 1.

    return precision, recall


# def average_precision(rank):
#     '''
#     Score is average precision (area under PR curve)
#     Relevance is binary (nonzero is relevant).
#     >>> rank = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
#     >>> delta_rank = 1. / sum(rank)
#     >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_rank for x, y in enumerate(rank) if y])
#     0.7833333333333333
#     >>> average_precision(r)
#     0.78333333333333333
#
#     Args:
#         r: Relevance scores (list or numpy) in rank order
#             (first element is the first item)
#
#     Returns:
#         Average precision
#     '''
#
#     rank = np.asarray(rank) != 0
#     out = [precision_at_k(rank, k + 1) for k in range(rank.size) if rank[k]]
#
#     if not out: return 0.
#
#     return np.mean(out)
#
#
# def mean_average_precision(rs):
#     """Score is mean average precision
#     Relevance is binary (nonzero is relevant).
#     >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
#     >>> mean_average_precision(rs)
#     0.78333333333333333
#     >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
#     >>> mean_average_precision(rs)
#     0.39166666666666666
#
#     Args:
#         rs: Iterator of relevance scores (list or numpy) in rank order
#             (first element is the first item)
#
#     Returns:
#         Mean average precision
#     """
#     return np.mean([average_precision(r) for r in rs])


def mae(y_true, y_pred):
    """Computes the Mean Absolute Error (MAE).

    :param y_true: Array of true ratings
    :param y_pred: Array of predicted ratings
    :return: Mean Absolute Error
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError('`true_score` and `pred_score` must have the exactly same shape.')

    return np.sum(np.fabs(y_true - y_pred)) / float(y_true.shape[0])


def rmse(y_true, y_pred):
    """
    Computes the Root Mean Squared Error (RMSE).

    Args:
        y_true: Array of true ratings
        y_pred: Array of predicted ratings

    Returns:
         Root Mean Squared Error
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError('`y_true` and `y_pred` must have the exactly same shape.')

    return (np.sum((y_true - y_pred) ** 2) / float(y_true.shape[0])) ** 0.5


def main():
    # obs.: the 'rank' numpy array is composed of desc-sorted indexes of the recommendation list

    # we have 5 items (A, B, C, D, E) and the relevance is binary

    # rank ground truth: (D, A, E, B, C)
    rank_true = np.array([3, 2, 2, 1, 0])

    # predicted rankings
    # (A, D, E, B, C)
    rank_pred_1 = np.array([2, 3, 2, 1, 0])

    # (E, B, A, D, C)
    rank_pred_2 = np.array([2, 1, 2, 3, 0])

    # (C, D, B, A, E)
    rank_pred_3 = np.array([0, 3, 1, 2, 2])

    k = rank_true.size
    form = "trad"

    print(f'Top {k} items')
    print(f'IDCG: {_dcg_score(rank_true, k=k, form=form)}')

    print(f'\nDCG pred 1: {_dcg_score(rank_pred_1, k=k, form=form)}')
    print(f'NDCG pred 1: {ndcg_score(rank_true, rank_pred_1, k=k, form=form)}')

    print(f'\nDCG pred 2: {_dcg_score(rank_pred_2, k=k, form=form)}')
    print(f'NDCG pred 2: {ndcg_score(rank_true, rank_pred_2, k=k, form=form)}')

    print(f'\nDCG pred 3: {_dcg_score(rank_pred_3, k=k, form=form)}')
    print(f'NDCG pred 3: {ndcg_score(rank_true, rank_pred_3, k=k, form=form)}')

    y_true = np.asfarray([5., 0, 0, 0, 4, 3, 5, 4, 0, 0])
    idx_y_true_sorted = np.argsort(y_true)[::-1]

    y_pred = np.asfarray([4.2, 1.5, 0., 4., 0.5, 4.8, 3.7, 0, 2.9, 0])
    idx_y_pred_sorted = np.argsort(y_pred)[::-1]

    print(f'y_true: {y_true} | idx_y_true_sorted: {idx_y_true_sorted}')
    print(f'y_pred: {y_pred} | idx_y_pred_sorted: {idx_y_pred_sorted}')

    print(f'y_true ranked: {y_true[idx_y_true_sorted]}')
    print(f'y_pred ranked: {y_true[idx_y_pred_sorted]}')

    for k in range(1, y_true.size + 1):
        ndcg = ndcg_score(y_true[idx_y_true_sorted], y_true[idx_y_pred_sorted], k=k, form='alt')
        print(f'NDCG@{k} for y_pred: {ndcg}')


if __name__ == "__main__": main()
