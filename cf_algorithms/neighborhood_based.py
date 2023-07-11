import time
import tqdm
import pickle as p
import numpy as np
import pandas as pd
from cf_algorithms import similarity as s


class UserKnn:

    def __init__(self, n_neighbors=1, similarity='pearson', pred_function='mean_centered'):

        self.similarity_name = 'custom'
        self.sim_function = similarity
        self.sim_interval = (-1., 1.)
        self.k = n_neighbors
        self.X = None
        self.user_ids = {}
        self.item_ids = {}
        self.train_indices = []
        self.trained = False
        self.pred_function = pred_function

        # if load_file:
        #     self.W = np.load('user_knn_similarity_matrix.npy')
        #
        #     with open('dict_mapped_users.bin', 'rb') as file:
        #         self.user_ids = p.load(file)
        #
        #     with open('dict_mapped_items.bin', 'rb') as file:
        #         self.item_ids = p.load(file)
        #
        #     self.trained = True

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, k):

        if not isinstance(k, (int, np.int32, np.int64)):
            raise TypeError('´k´ must be an integer.')

        if k < 1:
            raise ValueError('´k´ should be a strictly positive integer.')

        self._k = k

    @property
    def sim_function(self):
        return self._sim_function

    @sim_function.setter
    def sim_function(self, similarity):

        if similarity in s.SIMILARITIES.keys():
            self.similarity_name = similarity
            self._sim_function = s.SIMILARITIES[similarity]
            self.sim_interval = s.SIMILARITY_INTERVAL[similarity]

        elif callable(similarity):
            self.similarity_name = 'callable'
            self._sim_function = similarity

        else:
            raise TypeError('`similarity` must be a valid key function or a callable.')

    @property
    def pred_function(self):
        return self._pred_function

    @pred_function.setter
    def pred_function(self, function):

        if function == 'weighted_average':
            self._pred_function = self._weighted_average
        elif function == 'mean_centered':
            self._pred_function = self._mean_centered_weighted_average
        elif function == 'z_score':
            self._pred_function = self._z_score_weighted_average
        elif callable(function):
            self._pred_function = function
        else:
            raise TypeError('`pred_function` must be a valid string key or a callable.')

    def fit(self, X: pd.DataFrame, save_file: bool = False):

        print("UserKNN - Number of neighbors:", self.k)
        print("UserKNN - Similarity function:", self.similarity_name)

        self.X = self._transform(X)

        if self.k > self.X.shape[0]:
            self.k = int(self.X.shape[0]) - 1

        # Compute user-user similarities
        # self._compute_similarity()

        if save_file:
            # np.save('user_knn_similarity_matrix', self.W, allow_pickle=False)
            with open('dict_mapped_users.bin', 'wb') as file:
                p.dump(self.user_ids, file)

            with open('dict_mapped_items.bin', 'wb') as file:
                p.dump(self.item_ids, file)

        self.trained = True
        print("UserKNN - The model has been fitted.")

    def predict(self, X_val, k=None):

        if not isinstance(X_val, dict):
            raise TypeError('`X_val` must a dictionary[int: list].')

        if k is None:
            k = self.k

        if not self.trained:
            raise RuntimeError('UserKnn` should be trained before predicting on new test instances.')

        print("UserKNN - Predicting on test users...")

        start = time.time()

        X_pred = {}
        # for user_idx, items_list in X_val.items():
        for user_idx, items_list in tqdm.tqdm(X_val.items()):

            neighbors, W = self.query(user_idx, k=k)

            X_pred[user_idx] = [
                self._pred_function(user_idx, item_idx, neighbors, W)
                for item_idx in items_list]

        end = time.time()

        print(f"UserKNN - prediction time: {end - start:.2f} seconds.")

        return X_pred

    def query(self, user_idx, k=1):

        # Create user feature array
        x = self._user_vector(user_idx)

        if len(self.train_indices) == 0:

            n = len(self.user_ids)

            # Compute user-user similarities taking all users
            w = np.array([
                self.sim_function(x, self._user_vector(j))
                for j in self.user_ids
            ])

            # Avoid self-similarity during reverse sorting
            w[user_idx] = self.sim_interval[0]

        else:

            n = len(self.train_indices)

            # Compute user-user similarities taking only training users
            w = np.array([
                self.sim_function(x, self._user_vector(j))
                for j in self.train_indices
            ])

        idx_top_k = np.argpartition(w, n - k)[n - k:]
        sorted_idx = np.argsort(w[idx_top_k])[::-1]

        return idx_top_k[sorted_idx], w[idx_top_k[sorted_idx]]

    def query2d(self, w, k=1, return_similarities=True):

        # w must be a valid user index
        assert w in list(self.user_ids.keys())

        # Number of neighbors must be in the interval [1, n_users]
        assert 1 <= k <= self.W.shape[0]

        n = self.W.shape[0]
        sample_range = np.arange(n)[:, None]

        # Forces the partition and sorting to ignore self-similarities (minimum similarity)
        self.W.flat[:: n + 1] = self.sim_function.get('min_max')[0]

        # Arg partition considers descending order of values
        top_k_idx = np.argpartition(self.W, n - k, axis=1)[:, n - k:]

        # Arg sort considers descending order of values
        sorted_idx = np.argsort(self.W[sample_range, top_k_idx])[:, ::-1]

        # Reestablish self-similarities (maximum similarity)
        self.W.flat[:: n + 1] = self.sim_function.get('min_max')[1]

        if return_similarities:
            result = (top_k_idx[sample_range, sorted_idx][w],
                      self.W[sample_range, top_k_idx[sample_range, sorted_idx]][w])
        else:
            result = top_k_idx[sample_range, sorted_idx][w]

        return result

    def _transform(self, raw_data: pd.DataFrame):

        # List of column names
        columns = list(raw_data.columns)

        # Map raw user ids to a discrete integer interval
        self.user_ids = {i: x for i, x in enumerate(raw_data[columns[0]].unique())}
        user_raw_ids = {x: i for i, x in enumerate(raw_data[columns[0]].unique())}

        # Map raw movie ids to a discrete integer interval
        self.item_ids = {i: x for i, x in enumerate(raw_data[columns[1]].unique())}
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

        return X

    def _user_vector(self, user_idx):

        queried_user = self.X[self.X['user_id'] == user_idx][['item_id', 'rating']]
        indices = queried_user.iloc[:, 0].values.astype(np.int32)

        x = np.zeros(len(self.item_ids), dtype=np.float64)
        x[indices] = queried_user.iloc[:, 1].values.astype(np.float64)

        return x

    def _rating_score(self, user_idx, item_idx):

        q = f"user_id == {user_idx} and item_id == {item_idx}"

        return self.X.query(q).iloc[0, 2]

    def _user_mean_std(self, user_idx, excluded_items=[]):

        q = f"user_id == {user_idx} and item_id not in {excluded_items}"
        x = self.X.query(q)['rating']

        return x.mean(), x.std()

    def _check_common_neighbors(self, item_idx, neighbors):

        q = f"user_id in {list(neighbors)} and item_id == {item_idx}"

        return self.X.query(q)['user_id'].values.tolist()

    def _weighted_average(self, user_idx, item_idx, neighbors, W):

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

    def _mean_centered_weighted_average(self, user_idx, item_idx, neighbors, W):

        # Average rating of user i
        mu_i, _ = self._user_mean_std(user_idx, excluded_items=[item_idx])

        # filter valid neighbors (the ones who rated item i)
        valid_neighbors = self._check_common_neighbors(item_idx, neighbors)

        if not len(valid_neighbors):
            return mu_i

        sum_w = 0.
        norm_factor = 0. + 1e-5

        # Sum over all valid neighbors of user i
        for j in valid_neighbors:

            mu_j, _ = self._user_mean_std(j)
            r_j = self._rating_score(j, item_idx)
            sum_w += W[neighbors == j][0] * (r_j - mu_j)
            norm_factor += np.fabs(W[neighbors == j][0])

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

    def _compute_similarity(self):

        # Get the number of users (n)
        n = len(self.user_ids)

        print(f"Computing similarities with {self.sim_function.get('name').upper()}...")

        self.W = np.empty((n, n))

        # Guarantees the diagonal array matches maximum possible similarity
        self.W.flat[:: n + 1] = self.sim_function.get('min_max')[1]

        for u in self.user_ids.keys():

            x = self._user_vector(u)

            for v in self.user_ids.keys():

                y = self._user_vector(v)

                if u != v:
                    self.W[u, v] = self.sim_function.get('function')(x, y)

        print('Done.')