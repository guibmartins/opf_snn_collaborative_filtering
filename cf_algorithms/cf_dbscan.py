import tqdm
import numpy as np
import opfython.math.cf_distances as d
# import opfython.math.distance as d

NOISE = -1


class DBSCAN:

    def __init__(self, eps=1., min_samples=2, distance_function='euclidean', custom_state=None, verbose=True):

        self.eps = eps
        self.min_samples = min_samples
        self.distance = 'custom'
        self.distance_function = distance_function
        self.custom_state = custom_state
        self.trained = False
        self.labels = None
        self.seeds = []
        self.idx_seeds = []
        self._X = np.array([])
        self.verbose = verbose

    @property
    def eps(self):

        return self._eps

    @eps.setter
    def eps(self, eps):

        if not isinstance(eps, (float, np.float32, np.float64)):
            raise TypeError('´eps´ must be a float.')

        self._eps = eps

    @property
    def min_samples(self):

        return self._min_samples

    @min_samples.setter
    def min_samples(self, min_samples):

        if not isinstance(min_samples, (int, np.int32, np.int64)):

            raise TypeError('´min_samples´ must be an integer.')

        if min_samples <= 1:

            raise ValueError('´min_samples´ must be an integer greater than 1.')

        self._min_samples = min_samples

    @property
    def distance_function(self):
        return self._distance_function

    @distance_function.setter
    def distance_function(self, distance):

        if distance in d.DISTANCES.keys():

            self.distance = distance
            self._distance_function = d.DISTANCES[distance]

        elif distance == 'precomputed':
            self._distance_function = None

        elif callable(distance):

            self.distance = 'callable'
            self._distance_function = distance

        else:
            raise TypeError('`distance_function` must be a valid key function or a callable.')

    @property
    def seeds(self):

        return self.X[self.idx_seeds]

    @seeds.setter
    def seeds(self, seeds):

        self._seeds = seeds

    @property
    def X(self):

        return self._X

    @X.setter
    def X(self, X):

        if not isinstance(X, np.ndarray):

            raise TypeError('´X´ must be a numpy array.')

        self._X = X

    def fit(self, X_train):
        """
        Cluster array-based samples using the DBSCAN algorithm.

        dbscan takes a dataset `X_train` (a list of vectors), a threshold distance
        `eps`, and a required minimum number of points `min_samples`.

        It will return a list of cluster labels. The label -1 means noise, and then
        the clusters are numbered starting from 1.
        """

        # This list will hold the final cluster assignment for each sample in X_train.
        # There are two reserved values:
        #    -1 - Indicates a noise point
        #     0 - Means the point hasn't been considered yet.

        if self.verbose:
            print('Clustering with DBSCAN...')

        if not isinstance(X_train, np.ndarray):
            X_train = np.array(X_train)

        self.X = X_train

        # Initially all labels are 0.
        self.labels = np.zeros(X_train.shape[0], dtype=int)
        # labels = [0] * len(D)

        # 'current_cluster' is the ID of the current cluster.
        current_cluster = 0

        # This outer loop is just responsible for picking new seed points--a point
        # from which to grow a new cluster.
        # Once a valid seed point is found, a new cluster is created, and the
        # cluster growth is all handled by the 'expandCluster' routine.

        if self.verbose:
            # For each sample i in X_train...
            for i in tqdm.trange(X_train.shape[0]):

                # Only points that have not already been claimed
                # can be picked as new seed points.
                # If the point's label is not 0, continue to the next point.
                if not (self.labels[i] == 0):
                    continue

                # Find all of sample's i neighbors.
                neighbors = self._region_query(i)

                # If the number is below min_samples, this point is noise.
                # This is the only condition under which a point is labeled
                # NOISE -- when it's not a valid seed point. A NOISE point may later
                # be picked up by another cluster as a boundary point (this is the only
                # condition under which a cluster label can change -- from NOISE to
                # something else).
                if len(neighbors) < self.min_samples:
                    self.labels[i] = NOISE
                # Otherwise, if there are at least min_samples nearby,
                # use this point as the seed for a new cluster.
                else:
                    current_cluster += 1

                    self.idx_seeds.extend([i])
                    self._grow_cluster(i, neighbors, current_cluster)

        else:
            # For each sample i in X_train...
            for i in range(X_train.shape[0]):

                # Only points that have not already been claimed
                # can be picked as new seed points.
                # If the point's label is not 0, continue to the next point.
                if not (self.labels[i] == 0):
                    continue

                # Find all of sample's i neighbors.
                neighbors = self._region_query(i)

                # If the number is below min_samples, this point is noise.
                # This is the only condition under which a point is labeled
                # NOISE -- when it's not a valid seed point. A NOISE point may later
                # be picked up by another cluster as a boundary point (this is the only
                # condition under which a cluster label can change -- from NOISE to
                # something else).
                if len(neighbors) < self.min_samples:
                    self.labels[i] = NOISE
                # Otherwise, if there are at least min_samples nearby,
                # use this point as the seed for a new cluster.
                else:
                    current_cluster += 1

                    self.idx_seeds.extend([i])
                    self._grow_cluster(i, neighbors, current_cluster)

        # All data has been clustered!
        self.trained = True

    def predict(self, X_val, only_seeds=False):

        preds = np.ones(shape=X_val.shape[0], dtype=int) * NOISE

        # Predicting clusters using seed points only
        if only_seeds:

            for i in tqdm.trange(X_val.shape[0]):

                distances = [self.distance_function(X_val[i], self.seeds[j])
                             for j in range(len(self.seeds))]

                idx_min_dist = np.argmin(distances)

                if distances[idx_min_dist] < self.eps:

                    preds[i] = self.labels[self.idx_seeds[idx_min_dist]]

            return preds

        # Predicting clusters using the entire training data
        for i in tqdm.trange(X_val.shape[0]):

            distances = [self.distance_function(X_val[i], self.X[j])
                         for j in range(self.X.shape[0])]

            idx_min_dist = np.argmin(distances)

            if distances[idx_min_dist] < self.eps:

                preds[i] = self.labels[idx_min_dist]

        return preds

    def fit_predict(self, X_train):

        self.fit(X_train)

        return self.labels

    def _region_query(self, i):
        """
        Find all points in dataset `X_train` within distance `eps` of sample `i`.

        This function calculates the distance between a sample i and every other
        point in the dataset, and then returns only those points which are within a
        threshold distance `eps`.
        """
        x = self.X[i]
        neighbors = [j for j in range(self.X.shape[0])
                     if self.distance_function(x, self.X[j]) < self.eps]

        # # For each point in the dataset...
        # for j in range(self.X.shape[0]):
        #
        #     # If the distance is below the threshold, add it to the neighbors list.
        #     if np.linalg.norm(D[P] - D[Pn]) < eps:
        #         neighbors.append(Pn)

        return neighbors

    def _grow_cluster(self, i, neighbors, current_cluster):
        """
        Grow a new cluster with label `C` from the seed point `P`.

        This function searches through the dataset to find all points that belong
        to this new cluster. When this function returns, cluster `C` is complete.

        Parameters:
          `labels`          - List storing the cluster labels for all dataset points
          `i`               - Index of the seed point for this new cluster
          `neighbors`       - All neighbor of `i` sample
          `current_cluster` - The label for this new cluster.
        """

        # Assign the cluster label to the seed point.
        self.labels[i] = current_cluster

        # Look at each neighbor of i (neighbors are referred to as Pn).
        # 'neighbors' (list) will be used as a FIFO queue of points to search--that is, it
        # will grow as we discover new branch points for the cluster. The FIFO
        # behavior is accomplished by using a while-loop rather than a for-loop.
        # In NeighborPts, the points are represented by their index in the original
        # dataset.
        j = 0
        while j < len(neighbors):

            # Get the next sample (id) from the queue.
            k = neighbors[j]

            # If 'k' was labelled NOISE during the seed search, then we
            # know it's not a branch point (it doesn't have enough neighbors),
            # so make it a leaf point of cluster C and move on.
            if self.labels[k] == NOISE:

                self.labels[k] = current_cluster

            # Otherwise, if 'k' isn't already claimed, claim it as part of C.
            elif self.labels[k] == 0:

                # Add neighbor to cluster C (Assign cluster label C).
                self.labels[k] = current_cluster

                # Find all the neighbors of 'k'
                k_neighbors = self._region_query(k)

                # If 'k' has at least MinPts neighbors, it's a branch point!
                # Add all of its neighbors to the FIFO queue to be searched.
                if len(k_neighbors) >= self.min_samples:
                    neighbors.extend(k_neighbors)

                # If 'k' *doesn't* have enough neighbors, then it's a leaf point.
                # Don't queue up it's neighbors as expansion points.
                # else:
                # Do nothing
                # NeighborPts = NeighborPts

            # Advance to the next point in the FIFO queue.
            j += 1
