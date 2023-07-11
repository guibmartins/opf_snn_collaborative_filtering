import numpy as np
import tqdm
import opfython.math.cf_distances as d

EPSILON = 1e-20


class KMeans:

    def __init__(self, n_clusters, distance_function='euclidean', max_iter=100, custom_state=None):

        self.n_centers = n_clusters
        self.distance = 'custom'
        self.distance_function = distance_function
        self.max_iter = max_iter
        self.trained = False
        self.centroids = None
        self.custom_state = custom_state

    @property
    def n_centers(self):

        return self._n_centers

    @n_centers.setter
    def n_centers(self, n_centers):

        if not isinstance(n_centers, (int, np.int32, np.int64)):
            raise TypeError('´n_centers´ must be an integer.')

        if n_centers < 1:
            raise ValueError('´n_centers´ should be a strictly positive integer.')

        self._n_centers = n_centers

    @property
    def distance_function(self):
        return self._distance_function

    @distance_function.setter
    def distance_function(self, distance):

        if distance in d.DISTANCES.keys():

            self.distance = distance
            self._distance_function = d.DISTANCES[distance]

        elif callable(distance):

            self.distance = 'callable'
            self._distance_function = distance

        else:
            raise TypeError('`distance_function` must be a valid key function or a callable.')

    def _init_centers(self, X, custom_state=None):

        rstate = np.random.RandomState(custom_state)
        indices = rstate.permutation(X.shape[0])

        return X[indices[:self.n_centers], :]

    def fit(self, X_train):

        print("Clustering with K-Means...")

        if not isinstance(X_train, np.ndarray):
            X_train = np.array(X_train)

        # Initialize centroids
        centroids = self._init_centers(X_train, custom_state=self.custom_state)

        self.trained = False

        # Initialize distances array
        distances = np.empty(shape=(X_train.shape[0], centroids.shape[0]))

        # Compute distances between centroids and each training point
        for i in range(X_train.shape[0]):
            distances[i] = [self.distance_function(X_train[i], c) for c in centroids]

        # For each sample, get the index of the minimum distance among centroids
        cluster_labels = np.argmin(distances, axis=1)

        # Repeat the above steps until 'max_iter' iterations is reached
        # for _ in range(self.max_iter):
        for _ in tqdm.trange(self.max_iter):

            centroids = np.empty_like(centroids)

            for c in range(self.n_centers):

                # Samples from cluster c composes the subset X_c
                X_c = X_train[cluster_labels == c]

                if X_c.shape[0] > 0:
                    # Update centroids by taking the cluster's mean
                    centroids[c] = np.mean(X_c, axis=0)

            # Initialize distances array
            distances = np.empty(shape=(X_train.shape[0], centroids.shape[0]))

            # Compute distances between centroids and each training point
            for i in range(X_train.shape[0]):
                distances[i] = [self.distance_function(X_train[i], c) for c in centroids]

            # For each sample, get the index of the minimum distance among centroids
            cluster_labels = np.argmin(distances, axis=1)

        # Set the final centroid points
        self.centroids = centroids

        # The classifier has been fitted
        self.trained = True

        # return final cluster predictions for all training points
        return list(cluster_labels)

    def predict(self, X_val):

        if not self.trained:
            raise RuntimeError('The classifier has not been fitted yet.')

        # Initialize distances array
        distances = np.empty(shape=(X_val.shape[0], self.centroids.shape[0]))

        # Compute distances between centroids and each training point
        for i in tqdm.trange(X_val.shape[0]):
            distances[i] = [self.distance_function(X_val[i], c) for c in self.centroids]

        # For each sample, get the index of the minimum distance among centroids
        predictions = np.argmin(distances, axis=1)

        return list(predictions)
