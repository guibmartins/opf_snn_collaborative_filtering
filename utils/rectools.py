import math
import pandas as pd
import numpy as np
import utils.recmetrics as m
from scipy.sparse import csr_matrix


def train_test_split_wg(X: pd.DataFrame, item_size: float = 0.2, custom_state=None):
    """
    Train-Test split with respect only to items. This kind of split attends the so-called 'Weak Generalization
    Evaluation', i.e., both training and test sets may have the same users (items).
    :param X: pandas.DataFrame
    :param item_size: float
    :param custom_state: None or integer (int)
    :return: dictionary[user_id (int): test items (list)]
    """

    # Reset the seed generator
    # np.random.seed(custom_state)
    rstate = np.random.RandomState(custom_state)

    col_name = list(X.columns)
    users = X[col_name[0]].unique()

    # np.random.shuffle(users)
    rstate.shuffle(users)

    X_val = {}

    for user_idx in users:
        X_user = rstate.permutation(X[X[col_name[0]] == user_idx][col_name[1]])
        halt_items = math.ceil(X_user.shape[0] * (1 - item_size))
        X_val[user_idx] = list(X_user[halt_items:])

    return X_val


def train_test_split_sg(X: pd.DataFrame, user_size: float = .1, item_size: float = .1, custom_state=None):
    """
    Train-Test split with respect to users and items. This kind of split attends the Strong Generalization
    Evaluation, i.e., a subset of users and items will be strictly used as Test Set, such that a test user (item)
    will not be part of the Training Set.
    :param X: pandas.DataFrame
    :param user_size: float
    :param item_size: float
    :param custom_state: None or integer (int)
    :return: tuple [list, list, dictionary]
    """

    # Reset the seed generator
    rstate = np.random.RandomState(custom_state)

    col_name = list(X.columns)
    users = X[col_name[0]].unique()

    rstate.shuffle(users)

    halt_users = math.ceil(len(users) * (1 - user_size))
    idx_train = list(users[:halt_users])
    idx_val = list(users[halt_users:])

    X_val = {}
    for user_idx in users[halt_users:]:
        X_user = rstate.permutation(X[X[col_name[0]] == user_idx][col_name[1]])
        halt_items = math.ceil(X_user.shape[0] * (1 - item_size))
        X_val[user_idx] = list(X_user[halt_items:])

    return idx_train, idx_val, X_val


def transform(raw_data: pd.DataFrame):

    # List of column names
    columns = list(raw_data.columns)

    # Map raw user ids to a discrete integer interval
    user_ids = {i: x for i, x in enumerate(raw_data[columns[0]].unique())}
    user_raw_ids = {x: i for i, x in enumerate(raw_data[columns[0]].unique())}

    # Map raw movie ids to a discrete integer interval
    item_ids = {i: x for i, x in enumerate(raw_data[columns[1]].unique())}
    item_raw_ids = {x: i for i, x in enumerate(raw_data[columns[1]].unique())}

    X = pd.DataFrame()

    # Set inner user ids over raw user ids
    X[['user_id']] = raw_data[[columns[0]]].applymap(
        lambda x: user_raw_ids.get(x)).astype(np.int32)

    # Set inner item ids over raw item ids
    X[['item_id']] = raw_data[[columns[1]]].applymap(
        lambda x: item_raw_ids.get(x)).astype(np.int32)

    # Set rating scores as np.float64
    X[['rating']] = raw_data[[columns[2]]].astype(np.float64)

    return X, user_ids, item_ids


def create_cf_matrix(df: pd.DataFrame, size=None):
    n_users, n_items = size

    if size is None:
        n_users = df['user_id'].unique().shape[0]
        n_items = df['item_id'].unique().shape[0]

    X, user_mapping, item_mapping = transform(df)

    idx_users = X['user_id'].values.squeeze()
    idx_items = X['item_id'].values.squeeze()
    ratings = X['rating'].values.squeeze()

    # User-based sparse rating matrix
    R = csr_matrix(
        (ratings, (idx_users, idx_items)),
        shape=(n_users, n_items), dtype=np.float64)

    return R, X, user_mapping, item_mapping


def _check_common_neighbors(R_train, item_idx, neighbors):
    q = f"user_id in {list(neighbors)} and item_id == {item_idx}"
    return R_train.query(q)['user_id'].values.tolist()


def _rating_score(R_train, user_idx, item_idx):
    return R_train.query(f"user_id == {user_idx} and item_id == {item_idx}").iloc[0, 2]


def _user_mean_std(R_train, user_idx, excluded_items=[]):
    x = R_train.query(f"user_id == {user_idx} and item_id not in {excluded_items}")['rating']
    return x.mean(), x.std()


def _weighted_average(self, item_idx, neighbors, W):
    # filter valid neighbors (the ones who rated item i)
    valid_neighbors = self._check_common_neighbors(item_idx, neighbors)

    if not len(valid_neighbors):
        return 0.

    sum_w = 0
    norm_factor = 0.

    # Sum over all valid neighbors of user i
    for j in valid_neighbors:
        r_j = self._rating_score(j, item_idx)
        sum_w += W[neighbors == j][0] * r_j
        norm_factor += np.fabs(W[neighbors == j][0])

    return sum_w / norm_factor


def _mean_centered_weighted_average(R_train, user_idx, item_idx, neighbors, W):
    # Average rating of user i
    mu_i, _ = _user_mean_std(R_train, user_idx, excluded_items=[item_idx])

    # filter valid neighbors (the ones who rated item i)
    valid_neighbors = _check_common_neighbors(R_train, item_idx, neighbors)

    if not len(valid_neighbors):
        return mu_i

    sum_w = 0
    norm_factor = 0.

    valid_indices = [i for i, adj in enumerate(neighbors) if adj in valid_neighbors]

    # Sum over all valid neighbors of user i
    for j in valid_indices:
        mu_j, _ = _user_mean_std(R_train, neighbors[j])
        r_j = _rating_score(R_train, neighbors[j], item_idx)
        sum_w += W[j] * (r_j - mu_j)
        norm_factor += np.abs(W[j])

    if norm_factor == 0:
        return mu_i

    return mu_i + (sum_w / norm_factor)


def _z_score_weighted_average(self, user_idx, item_idx, neighbors, W):
    # Mean and standard deviation of user i observed ratings
    mu_i, std_i = self._user_mean_std(user_idx, excluded_items=[item_idx])

    # filter valid neighbors (the ones who rated item i)
    valid_neighbors = self._check_common_neighbors(item_idx, neighbors)

    if not len(valid_neighbors):
        return mu_i

    sum_w = 0
    norm_factor = 0.

    # Sum over all valid neighbors of user i
    for j in valid_neighbors:
        mu_j, std_j = self._user_mean_std(j)
        z_j = (self._rating_score(j, item_idx) - mu_j) / std_j
        sum_w += W[neighbors == j][0] * z_j
        norm_factor += np.fabs(W[neighbors == j][0])

    return mu_i + (std_i * (sum_w / norm_factor))


def get_true_ratings(X: pd.DataFrame, X_val: dict):
    X_true = {}

    for u, items in X_val.items():
        X_true[u] = [X.query(f"user_id == {u} and item_id =={i}").iloc[0, 2] for i in items]

    return X_true


def get_average_ratings(X: pd.DataFrame, X_val: dict, baseline=None):
    if baseline is not None and \
            isinstance(baseline, (float, np.float32, np.float64)):
        return {u: baseline for u in X_val.keys()}

    idx_val = list(X_val.keys())
    X_mean = X.query(f"user_id in {idx_val}")[['user_id', 'rating']].groupby("user_id").mean()

    return {u: X_mean.loc[u][0] for u in X_val.keys()}


def evaluate_prediction_task(X_true: dict, X_pred: dict, verbose=False):
    N = float(len(X_true.keys()))
    mae = rmse = 0.

    for u in X_pred.keys():
        x_true = np.array(X_true[u])
        x_pred = np.array(X_pred[u])

        # Calculate MAE regarding user u
        mae += m.mae(x_true, x_pred)

        # Calculate RMSE regarding user u
        rmse += m.rmse(x_true, x_pred)

    mae /= N
    rmse /= N

    if verbose:
        print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f}")

    return mae, rmse


def evaluate_ranking_task(X_true: dict, X_pred: dict, X_rel: dict, k=1, verbose=False):
    N = float(len(X_true.keys()))

    ndcg = 0.

    for u in X_pred.keys():

        x_true = np.array(X_true[u])
        x_pred = np.array(X_pred[u])

        # Ranking both the ground truth and the predictions
        idx_rank_true = np.argsort(x_true)[::-1]
        idx_rank_pred = np.argsort(x_pred)[::-1]

        # Get the ranked arrays
        rank_true = (x_true[idx_rank_true] > X_rel[u]).astype(int)
        rank_pred = (x_true[idx_rank_pred] > X_rel[u]).astype(int)

        if len(x_pred) < k:
            N -= 1
            continue

        # Calculate NDCG@k regarding user u
        ndcg += m.ndcg_score(rank_true, rank_pred, k=k, form='alt')

    ndcg /= N

    if verbose:
        print(f"NDCG@{k}: {ndcg:.4f}")

    return ndcg


def evaluate_recommendation_quality(X_true: dict, X_pred: dict, X_rel: dict, k=1, verbose=False):
    N = float(len(X_true.keys()))

    precision = recall = 0.

    for u in X_pred.keys():

        x_true = np.array(X_true[u])
        x_pred = np.array(X_pred[u])

        p, r = m.precision_recall_at_k(x_true, x_pred, k=k, relevance=X_rel[u])

        if len(x_pred) < k:
            N -= 1
            continue

        precision += p
        recall += r

    precision /= N
    recall /= N

    if verbose:
        print(f"Precision@{k}: {precision:.4f} | Recall@{k:} {recall:.4f}")

    return precision, recall


aggregation_functions = {
    'weighted': _weighted_average,
    'mean_centered': _mean_centered_weighted_average,
    'z_score': None
}
