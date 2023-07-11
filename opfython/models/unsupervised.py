"""Unsupervised Optimum-Path Forest.
"""

import time

import tqdm
import numpy as np

import opfython.utils.constants as c
import opfython.utils.exception as e
import opfython.utils.logging as log
from opfython.core import OPF, Heap
from opfython.subgraphs import KNNSubgraph
from opfython.subgraphs import NewKNNSubgraph

logger = log.get_logger(__name__)


class UnsupervisedOPF(OPF):
    """An UnsupervisedOPF which implements the unsupervised version of OPF classifier.

    References:
        L. M. Rocha, F. A. M. Cappabianco, A. X. Falcão.
        Data clustering as an optimum-path forest problem with applications in image analysis.
        International Journal of Imaging Systems and Technology (2009).

    """

    def __init__(self, min_k=1, max_k=1, distance='log_squared_euclidean', pre_computed_distance=None):
        """Initialization method.

        Args:
            min_k (int): Minimum `k` value for cutting the subgraph.
            max_k (int): Maximum `k` value for cutting the subgraph.
            distance (str): An indicator of the distance metric to be used.
            pre_computed_distance (str): A pre-computed distance file for feeding into OPF.

        """

        logger.info('Overriding class: OPF -> UnsupervisedOPF.')

        super(UnsupervisedOPF, self).__init__(distance, pre_computed_distance)

        # Defining the minimum `k` value for cutting the subgraph
        self.min_k = min_k

        # Defining the maximum `k` value for cutting the subgraph
        self.max_k = max_k

        logger.info('Class overrided.')

    @property
    def min_k(self):
        """int: Minimum `k` value for cutting the subgraph.

        """

        return self._min_k

    @min_k.setter
    def min_k(self, min_k):
        if not isinstance(min_k, int):
            raise e.TypeError('`min_k` should be an integer')
        if min_k < 1:
            raise e.ValueError('`min_k` should be >= 1')

        self._min_k = min_k

    @property
    def max_k(self):
        """int: Maximum `k` value for cutting the subgraph.

        """

        return self._max_k

    @max_k.setter
    def max_k(self, max_k):
        if not isinstance(max_k, int):
            raise e.TypeError('`max_k` should be an integer')
        if max_k < 1:
            raise e.ValueError('`max_k` should be >= 1')
        if max_k < self.min_k:
            raise e.ValueError('`max_k` should be >= `min_k`')

        self._max_k = max_k

    def _clustering(self, n_neighbours):
        """Clusters the subgraph using using a `k` value (number of neighbours).

        Args:
            n_neighbours (int): Number of neighbours to be used.

        """

        for i in range(self.subgraph.n_nodes):
            for k in range(n_neighbours):
                # Gathers node `i` adjacent node
                j = int(self.subgraph.nodes[i].adjacency[k])

                # If both nodes' density are equal
                if self.subgraph.nodes[i].density == self.subgraph.nodes[j].density:
                    # Turns on the insertion flag
                    insert = True

                    # For every possible `l` value
                    for l in range(n_neighbours):
                        # Gathers node `j` adjacent node
                        adj = int(self.subgraph.nodes[j].adjacency[l])

                        # If the nodes are the same
                        if i == adj:
                            # Turns off the insertion flag
                            insert = False

                        # If it is supposed to be inserted
                        if insert:
                            # Inserts node `i` in the adjacency list of `j`
                            self.subgraph.nodes[j].adjacency.insert(0, i)

                            # Increments the amount of adjacent nodes
                            self.subgraph.nodes[j].n_plateaus += 1

        # Creating a maximum heap
        h = Heap(size=self.subgraph.n_nodes, policy='max')

        for i in range(self.subgraph.n_nodes):
            # Updates the node's cost on the heap
            h.cost[i] = self.subgraph.nodes[i].cost

            # Defines node's `i` predecessor as NIL
            self.subgraph.nodes[i].pred = c.NIL

            # And its root as its same identifier
            self.subgraph.nodes[i].root = i

            # Inserts the node in the heap
            h.insert(i)

        # Defining an `l` counter
        l = 0

        while not h.is_empty():
            # Removes a node
            p = h.remove()

            # Appends its index to the ordered list
            self.subgraph.idx_nodes.append(p)

            # If the node's predecessor is NIL
            if self.subgraph.nodes[p].pred == c.NIL:
                # Updates its cost on the heap
                h.cost[p] = self.subgraph.nodes[p].density

                # Defines its cluster label as `l`
                self.subgraph.nodes[p].cluster_label = l

                # Increments the cluster identifier
                l += 1

            # Apply current node's cost as the heap's cost
            self.subgraph.nodes[p].cost = h.cost[p]

            # Calculates the number of its adjacent nodes
            n_adjacents = self.subgraph.nodes[p].n_plateaus + n_neighbours

            # For every possible adjacent node
            for k in range(n_adjacents):
                # Gathers the adjacent identifier
                q = int(self.subgraph.nodes[p].adjacency[k])

                if h.color[q] != c.BLACK:
                    # Calculates the current cost
                    current_cost = np.minimum(h.cost[p], self.subgraph.nodes[q].density)

                    # If temporary cost is bigger than heap's cost
                    if current_cost > h.cost[q]:
                        # Apply `q` predecessor as `p`
                        self.subgraph.nodes[q].pred = p

                        # Gathers the same root's identifier
                        self.subgraph.nodes[q].root = self.subgraph.nodes[p].root

                        # And its cluster label
                        self.subgraph.nodes[q].cluster_label = self.subgraph.nodes[p].cluster_label

                        # Updates the heap `q` node and the current cost
                        h.update(q, current_cost)

        # The final number of clusters will be equal to `l`
        self.subgraph.n_clusters = l

    def _normalized_cut(self, n_neighbours):
        """Performs a normalized cut over the subgraph using a `k` value (number of neighbours).

        Args:
            n_neighbours (int): Number of neighbours to be used.

        Returns:
            The value of the normalized cut.

        """

        # Defining an array to represent the internal cluster distances
        internal_cluster = np.zeros(self.subgraph.n_clusters)

        # Defining an array to represent the external cluster distances
        external_cluster = np.zeros(self.subgraph.n_clusters)

        # Defining the cut value
        cut = 0.0

        for i in range(self.subgraph.n_nodes):
            # Calculates its number of adjacent nodes
            n_adjacents = self.subgraph.nodes[i].n_plateaus + n_neighbours

            for k in range(n_adjacents):
                # Gathers its adjacent node identifier
                j = int(self.subgraph.nodes[i].adjacency[k])

                if self.pre_computed_distance:
                    distance = self.pre_distances[self.subgraph.nodes[i].idx][self.subgraph.nodes[j].idx]

                else:
                    distance = self.distance_fn(self.subgraph.nodes[i].features, self.subgraph.nodes[j].features)

                if distance > 0.0:
                    # If nodes belongs to the same clusters
                    if self.subgraph.nodes[i].cluster_label == self.subgraph.nodes[j].cluster_label:
                        # Increments the internal cluster distance
                        internal_cluster[self.subgraph.nodes[i].cluster_label] += 1 / distance

                    # If nodes belongs to distinct clusters
                    else:
                        # Increments the external cluster distance
                        external_cluster[self.subgraph.nodes[i].cluster_label] += 1 / distance

        for l in range(self.subgraph.n_clusters):
            # If the sum of internal and external clusters is bigger than 0
            if internal_cluster[l] + external_cluster[l] > 0.0:
                # Increments the value of the cut
                cut += external_cluster[l] / \
                    (internal_cluster[l] + external_cluster[l])

        return cut

    def _best_minimum_cut(self, min_k, max_k):
        """Performs a minimum cut on the subgraph using the best `k` value.

        Args:
            min_k (int): Minimum value of k.
            max_k (int): Maximum value of k.

        """

        logger.debug('Calculating the best minimum cut within [%d, %d] ...', min_k, max_k)

        # Calculates the maximum possible distances
        max_distances = self.subgraph.create_arcs(
            max_k, self.distance_fn, self.pre_computed_distance, self.pre_distances)

        # Initialize the minimum cut as maximum possible value
        min_cut = c.FLOAT_MAX

        best_k = self.max_k

        for k in range(min_k, max_k + 1):
            # If minimum cut is different than zero
            if min_cut != 0.0:
                # Gathers the subgraph's density
                self.subgraph.density = max_distances[k - 1]

                # Gathers current `k` as the subgraph's best `k` value
                self.subgraph.best_k = k

                # Calculates the p.d.f.
                self.subgraph.calculate_pdf(
                    k, self.distance_fn, self.pre_computed_distance, self.pre_distances)

                # Clustering with current `k` value
                self._clustering(k)

                # Performs the normalized cut with current `k` value
                cut = self._normalized_cut(k)

                if cut < min_cut:
                    min_cut = cut
                    best_k = k

        self.subgraph.destroy_arcs()

        # Applying best `k` value to the subgraph
        self.subgraph.best_k = best_k

        # Creating new arcs with the best `k` value
        self.subgraph.create_arcs(best_k, self.distance_fn, self.pre_computed_distance, self.pre_distances)

        # Calculating the new p.d.f. with the best `k` value
        self.subgraph.calculate_pdf(best_k, self.distance_fn, self.pre_computed_distance, self.pre_distances)

        logger.debug('Best: %d | Minimum cut: %.2f.', best_k, min_cut)

    def fit(self, X_train, Y_train=None, I_train=None):
        """Fits data in the classifier.

        Args:
            X_train (np.array): Array of training features.
            Y_train (np.array): Array of training labels.
            I_train (np.array): Array of training indexes.

        """

        logger.info('Clustering with classifier ...')

        start = time.time()

        # Creating a subgraph
        self.subgraph = KNNSubgraph(X_train, Y_train, I_train)

        # Performing the best minimum cut on the subgraph
        self._best_minimum_cut(self.min_k, self.max_k)

        # Clustering the data with best `k` value
        self._clustering(self.subgraph.best_k)

        # The subgraph has been properly trained
        self.subgraph.trained = True

        end = time.time()

        train_time = end - start

        logger.info('Classifier has been clustered with.')
        logger.info('Number of clusters: %d.', self.subgraph.n_clusters)
        logger.info('Clustering time: %s seconds.', train_time)

    def predict(self, X_val, I_val=None):
        """Predicts new data using the pre-trained classifier.

        Args:
            X_val (np.array): Array of validation features.
            I_val (np.array): Array of validation indexes.

        Returns:
            A list of predictions for each record of the data.

        """

        if not self.subgraph:
            raise e.BuildError('KNNSubgraph has not been properly created')

        if not self.subgraph.trained:
            raise e.BuildError('Classifier has not been properly clustered')

        logger.info('Predicting data ...')

        start = time.time()

        # Creating a prediction subgraph
        pred_subgraph = KNNSubgraph(X_val, I=I_val)

        # Gathering the best `k` value
        best_k = self.subgraph.best_k

        # Creating an array of distances
        distances = np.zeros(best_k + 1)

        # Creating an array of nearest neighbours indexes
        neighbours_idx = np.zeros(best_k + 1)

        for i in range(pred_subgraph.n_nodes):
            # Defines the current cost
            cost = -c.FLOAT_MAX

            # Filling array of distances with maximum value
            distances.fill(c.FLOAT_MAX)

            for j in range(self.subgraph.n_nodes):
                if j != i:
                    if self.pre_computed_distance:
                        distances[best_k] = self.pre_distances[pred_subgraph.nodes[i].idx][self.subgraph.nodes[j].idx]

                    else:
                        distances[best_k] = self.distance_fn(pred_subgraph.nodes[i].features, self.subgraph.nodes[j].features)

                    # Apply node `j` as a neighbour
                    neighbours_idx[best_k] = j

                    # Gathers current `k`
                    cur_k = best_k

                    # While current `k` is bigger than 0 and the `k` distance is smaller than `k-1` distance
                    while cur_k > 0 and distances[cur_k] < distances[cur_k - 1]:
                        # Swaps the distance from `k` and `k-1`
                        distances[cur_k], distances[cur_k -
                                                    1] = distances[cur_k - 1], distances[cur_k]

                        # Swaps the neighbours indexex from `k` and `k-1`
                        neighbours_idx[cur_k], neighbours_idx[cur_k -
                                                              1] = neighbours_idx[cur_k - 1], neighbours_idx[cur_k]

                        # Decrements `k`
                        cur_k -= 1

            # Defining the density as 0
            density = 0.0

            for k in range(best_k):
                density += np.exp(-distances[k] / self.subgraph.constant)

            density /= best_k

            # Scale the density between minimum and maximum values
            density = ((c.MAX_DENSITY - 1) * (density - self.subgraph.min_density) /
                       (self.subgraph.max_density - self.subgraph.min_density + c.EPSILON)) + 1

            for k in range(best_k):
                if distances[k] != c.FLOAT_MAX:
                    # Gathers the node's neighbour
                    neighbour = int(neighbours_idx[k])

                    # Calculate the temporary cost
                    temp_cost = np.minimum(self.subgraph.nodes[neighbour].cost, density)

                    # If temporary cost is bigger than current cost
                    if temp_cost > cost:
                        # Replaces the current cost
                        cost = temp_cost

                        # Propagates the predicted label from the neighbour
                        pred_subgraph.nodes[i].predicted_label = self.subgraph.nodes[neighbour].predicted_label

                        # Propagates the cluster label from the neighbour
                        pred_subgraph.nodes[i].cluster_label = self.subgraph.nodes[neighbour].cluster_label

                        pred_subgraph.nodes[i].density = density

                        pred_subgraph.nodes[i].cost = self.subgraph.nodes[neighbour].cost

        # Creating the list of predictions
        preds = [pred.predicted_label for pred in pred_subgraph.nodes]

        # Creating the list of clusters
        clusters = [pred.cluster_label for pred in pred_subgraph.nodes]

        end = time.time()

        predict_time = end - start

        logger.info('Data has been predicted.')
        logger.info('Prediction time: %s seconds.', predict_time)

        return preds, clusters, pred_subgraph

    def predict_cf(self, X_val, I_val=None):
        """Predicts new data using the pre-trained classifier.

        Args:
            X_val (np.array): Array of validation features.
            I_val (np.array): Array of validation indexes.

        Returns:
            A list of predictions for each record of the data.

        """

        if not self.subgraph:
            raise e.BuildError('KNNSubgraph has not been properly created')

        if not self.subgraph.trained:
            raise e.BuildError('Classifier has not been properly clustered')

        logger.info('Predicting data ...')

        start = time.time()

        # Creating a prediction subgraph
        pred_subgraph = KNNSubgraph(X_val, I=I_val)

        # Gathering the best `k` value
        best_k = self.subgraph.best_k

        # Creating an array of distances
        distances = np.zeros(best_k + 1)

        # Creating an array of nearest neighbours indexes
        neighbours_idx = np.zeros(best_k + 1)

        for i in range(pred_subgraph.n_nodes):
            # Defines the current cost
            cost = -c.FLOAT_MAX

            # Filling array of distances with maximum value
            distances.fill(c.FLOAT_MAX)

            for j in range(self.subgraph.n_nodes):
                if j != i:
                    if self.pre_computed_distance:
                        distances[best_k] = self.pre_distances[pred_subgraph.nodes[i].idx][self.subgraph.nodes[j].idx]

                    else:
                        distances[best_k] = self.distance_fn(pred_subgraph.nodes[i].features, self.subgraph.nodes[j].features)

                    # Apply node `j` as a neighbour
                    neighbours_idx[best_k] = j

                    # Gathers current `k`
                    cur_k = best_k

                    # While current `k` is bigger than 0 and the `k` distance is smaller than `k-1` distance
                    while cur_k > 0 and distances[cur_k] < distances[cur_k - 1]:
                        # Swaps the distance from `k` and `k-1`
                        distances[cur_k], distances[cur_k -
                                                    1] = distances[cur_k - 1], distances[cur_k]

                        # Swaps the neighbours indexex from `k` and `k-1`
                        neighbours_idx[cur_k], neighbours_idx[cur_k -
                                                              1] = neighbours_idx[cur_k - 1], neighbours_idx[cur_k]

                        # Decrements `k`
                        cur_k -= 1

            pred_subgraph.nodes[i].adjacency = neighbours_idx[:best_k].tolist()
            pred_subgraph.nodes[i].adj_distances = distances[:best_k].tolist()

            # Defining the density as 0
            density = 0.0

            for k in range(best_k):
                density += np.exp(-distances[k] / self.subgraph.constant)

            density /= best_k

            # Scale the density between minimum and maximum values
            density = ((c.MAX_DENSITY - 1) * (density - self.subgraph.min_density) /
                       (self.subgraph.max_density - self.subgraph.min_density + c.EPSILON)) + 1

            for k in range(best_k):
                if distances[k] != c.FLOAT_MAX:
                    # Gathers the node's neighbour
                    neighbour = int(neighbours_idx[k])

                    # Calculate the temporary cost
                    temp_cost = np.minimum(self.subgraph.nodes[neighbour].cost, density)

                    # If temporary cost is bigger than current cost
                    if temp_cost > cost:
                        # Replaces the current cost
                        cost = temp_cost

                        # Propagates the predicted label from the neighbour
                        pred_subgraph.nodes[i].predicted_label = self.subgraph.nodes[neighbour].predicted_label

                        # Propagates the cluster label from the neighbour
                        pred_subgraph.nodes[i].cluster_label = self.subgraph.nodes[neighbour].cluster_label

        # Creating the list of predictions
        preds = [pred.predicted_label for pred in pred_subgraph.nodes]

        # Creating the list of clusters
        clusters = [pred.cluster_label for pred in pred_subgraph.nodes]

        end = time.time()

        predict_time = end - start

        logger.info('Data has been predicted.')
        logger.info('Prediction time: %s seconds.', predict_time)

        return pred_subgraph, clusters

    def fit_predict(self, X_train, Y_train=None, I_train=None):

        self.fit(X_train, Y_train=Y_train, I_train=I_train)

        return np.array([node.cluster_label for node in self.subgraph.nodes])

    def propagate_labels(self):
        """Runs through the clusters and propagate the clusters roots labels to the samples.

        """

        logger.info('Assigning predicted labels from clusters ...')

        for i in range(self.subgraph.n_nodes):
            # Gathers the root from the node
            root = self.subgraph.nodes[i].root

            if root == i:
                self.subgraph.nodes[i].predicted_label = self.subgraph.nodes[i].label

            else:
                self.subgraph.nodes[i].predicted_label = self.subgraph.nodes[root].label

        logger.info('Labels assigned.')


class HierarchicalUnsupervisedOPF(UnsupervisedOPF):

    """An agglomerative-based hierarchical OPF clustering which
    implements the unsupervised version of OPF classifier.

    """
    def __init__(self, min_k=1, max_k=1, distance='squared_euclidean', pre_computed_distance=None):
        """Initialization method.

        Args:
            min_k (int): Minimum `k` value for cutting the subgraph.
            max_k (int): Maximum `k` value for cutting the subgraph.
            distance (str): An indicator of the distance metric to be used.
            pre_computed_distance (str): A pre-computed distance file for feeding into OPF.

        """
        # Override its parent class with the receiving arguments
        super(HierarchicalUnsupervisedOPF, self).__init__(min_k, max_k, distance, pre_computed_distance)

        self.sg_hierarchy = []

        self.clusters_hierarchy = []

        self.fit_time = 0.

        self.pred_time = 0.

    @property
    def sg_hierarchy(self):
        return self._sg_hierarchy

    @sg_hierarchy.setter
    def sg_hierarchy(self, sg_hierarchy):
        if not isinstance(sg_hierarchy, list):
            raise e.TypeError('`sg_hierarchy` should be a list.')

        self._sg_hierarchy = sg_hierarchy

    def fit(self, X_train, Y_train=None, I_train=None):
        """Fits data in the classifier.

        Args:
            X_train (np.array): Array of training features.
            Y_train (np.array): Array of training labels.
            I_train (np.array): Array of training indexes.

        """

        print('Clustering with Hierarchical OPF...')

        start = time.time()

        layer = 0
        max_k = self.max_k
        X_layer = np.copy(X_train)
        n_clusters = X_train.shape[0]

        while n_clusters > 1:

            print('Clustering on layer', layer, '...')

            opf = UnsupervisedOPF(max_k=max_k, distance=self.distance,
                                  pre_computed_distance=self.pre_computed_distance)

            clusters = opf.fit_predict(X_layer, Y_train, I_train)

            n_clusters = opf.subgraph.n_clusters

            # Stack prototypes to compose the training set with respect to layer l
            X_layer = np.asfarray([i.features
                                   for label in range(n_clusters)
                                   for i in opf.subgraph.nodes
                                   if (i.pred == -1 and i.cluster_label == label)])

            self.sg_hierarchy.append(opf)
            self.clusters_hierarchy.append(clusters)

            # Update `max_k`parameter considering the current clustering results
            max_k = int(np.min([opf.subgraph.best_k, X_layer.shape[0] - 1]))

            layer += 1

        self.fit_time = time.time() - start
        print(f'Clustering time: {self.fit_time:.4f} seconds.')

    def propagate_clusters(self, clusters_hierarchy: list):

        if len(clusters_hierarchy) == 1:
            return clusters_hierarchy.pop()

        # Get index of the last layer
        L = len(clusters_hierarchy) - 1

        print('Propagate labels from layer', L, 'to', L - 1, '...')

        preds = clusters_hierarchy[L - 1].copy()

        for i_prototype, cluster_id in enumerate(clusters_hierarchy[L]):
            idx = np.flatnonzero(clusters_hierarchy[L - 1] == i_prototype)
            preds[idx] = cluster_id

        clusters_hierarchy[L - 1] = preds
        clusters_hierarchy.pop()

        return self.propagate_clusters(clusters_hierarchy)


class ClusteringOPF(OPF):
    """An UnsupervisedOPF which implements the unsupervised version of OPF classifier.

    References:
        L. M. Rocha, F. A. M. Cappabianco, A. X. Falcão.
        Data clustering as an optimum-path forest problem with applications in image analysis.
        International Journal of Imaging Systems and Technology (2009).

    """

    def __init__(self, min_k=1, max_k=1, distance='log_squared_euclidean', pre_computed_distance=None):
        """Initialization method.

        Args:
            min_k (int): Minimum `k` value for cutting the subgraph.
            max_k (int): Maximum `k` value for cutting the subgraph.
            distance (str): An indicator of the distance metric to be used.
            pre_computed_distance (str): A pre-computed distance file for feeding into OPF.

        """

        logger.info('Overriding class: OPF -> UnsupervisedOPF.')

        super(ClusteringOPF, self).__init__(distance, pre_computed_distance)

        # Defining the minimum `k` value for cutting the subgraph
        self.min_k = min_k

        # Defining the maximum `k` value for cutting the subgraph
        self.max_k = max_k

        logger.info('Class overrided.')

    @property
    def min_k(self):
        """int: Minimum `k` value for cutting the subgraph.

        """

        return self._min_k

    @min_k.setter
    def min_k(self, min_k):
        if not isinstance(min_k, int):
            raise e.TypeError('`min_k` should be an integer')
        if min_k < 1:
            raise e.ValueError('`min_k` should be >= 1')

        self._min_k = min_k

    @property
    def max_k(self):
        """int: Maximum `k` value for cutting the subgraph.

        """

        return self._max_k

    @max_k.setter
    def max_k(self, max_k):
        if not isinstance(max_k, int):
            raise e.TypeError('`max_k` should be an integer')
        if max_k < 1:
            raise e.ValueError('`max_k` should be >= 1')
        if max_k < self.min_k:
            raise e.ValueError('`max_k` should be >= `min_k`')

        self._max_k = max_k

    def _clustering(self, n_neighbours):
        """Clusters the subgraph using a `k` value (number of neighbours).

        Args:
            n_neighbours (int): Number of neighbours to be used.

        """

        for i in range(self.subgraph.n_nodes):
            for k in range(n_neighbours):
                # Gathers node `i` adjacent node
                j = int(self.subgraph.nodes[i].adjacency[k])

                # If both nodes' density are equal
                if self.subgraph.nodes[i].density == self.subgraph.nodes[j].density:
                    # Turns on the insertion flag
                    insert = True

                    # For every possible `l` value
                    for l in range(n_neighbours):
                        # Gathers node `j` adjacent node
                        adj = int(self.subgraph.nodes[j].adjacency[l])

                        # If the nodes are the same
                        if i == adj:
                            # Turns off the insertion flag
                            insert = False

                        # If it is supposed to be inserted
                        if insert:
                            # Inserts node `i` in the adjacency list of `j`
                            self.subgraph.nodes[j].adjacency.insert(0, i)

                            # Increments the amount of adjacent nodes
                            self.subgraph.nodes[j].n_plateaus += 1

        # Creating a maximum heap
        h = Heap(size=self.subgraph.n_nodes, policy='max')

        # For every possible node
        for i in range(self.subgraph.n_nodes):

            # Updates the node's cost on the heap
            h.cost[i] = self.subgraph.nodes[i].cost

            # Defines node's `i` predecessor as NIL
            self.subgraph.nodes[i].pred = c.NIL

            # And its root as its same identifier
            self.subgraph.nodes[i].root = i

            # Inserts the node in the heap
            h.insert(i)

        # Defining an `l` counter
        l = 0

        while not h.is_empty():
            # Removes a node
            p = h.remove()

            # Appends its index to the ordered list
            self.subgraph.idx_nodes.append(p)

            # If the node's predecessor is NIL
            if self.subgraph.nodes[p].pred == c.NIL:
                # Updates its cost on the heap
                h.cost[p] = self.subgraph.nodes[p].density

                # Defines its cluster label as `l`
                self.subgraph.nodes[p].cluster_label = l

                # Increments the cluster identifier
                l += 1

            # Apply current node's cost as the heap's cost
            self.subgraph.nodes[p].cost = h.cost[p]

            # Calculates the number of its adjacent nodes
            n_adjacents = self.subgraph.nodes[p].n_plateaus + n_neighbours

            # For every possible adjacent node
            for k in range(n_adjacents):
                # Gathers the adjacent identifier
                q = int(self.subgraph.nodes[p].adjacency[k])

                if h.color[q] != c.BLACK:
                    # Calculates the current cost
                    current_cost = np.minimum(h.cost[p], self.subgraph.nodes[q].density)

                    # If temporary cost is bigger than heap's cost
                    if current_cost > h.cost[q]:
                        # Apply `q` predecessor as `p`
                        self.subgraph.nodes[q].pred = p

                        # Gathers the same root's identifier
                        self.subgraph.nodes[q].root = self.subgraph.nodes[p].root

                        # And its cluster label
                        self.subgraph.nodes[q].cluster_label = self.subgraph.nodes[p].cluster_label

                        # Updates the heap `q` node and the current cost
                        h.update(q, current_cost)

        # The final number of clusters will be equal to `l`
        self.subgraph.n_clusters = l

    def _normalized_cut(self, n_neighbours):
        """Performs a normalized cut over the subgraph using a `k` value (number of neighbours).

        Args:
            n_neighbours (int): Number of neighbours to be used.

        Returns:
            The value of the normalized cut.

        """

        # Defining an array to represent the internal cluster distances
        internal_cluster = np.zeros(self.subgraph.n_clusters)

        # Defining an array to represent the external cluster distances
        external_cluster = np.zeros(self.subgraph.n_clusters)

        # Defining the cut value
        cut = 0.0

        for i in range(self.subgraph.n_nodes):
            # Calculates its number of adjacent nodes
            n_adjacents = self.subgraph.nodes[i].n_plateaus + n_neighbours

            for k in range(n_adjacents):
                # Gathers its adjacent node identifier
                j = int(self.subgraph.nodes[i].adjacency[k])

                if self.pre_computed_distance:
                    distance = self.pre_distances[self.subgraph.nodes[i].idx][self.subgraph.nodes[j].idx]

                else:
                    distance = self.distance_fn(self.subgraph.nodes[i].features, self.subgraph.nodes[j].features)

                if distance > 0.0:
                    # If nodes belongs to the same clusters
                    if self.subgraph.nodes[i].cluster_label == self.subgraph.nodes[j].cluster_label:
                        # Increments the internal cluster distance
                        internal_cluster[self.subgraph.nodes[i].cluster_label] += 1. / distance

                    # If nodes belongs to distinct clusters
                    else:
                        # Increments the external cluster distance
                        external_cluster[self.subgraph.nodes[i].cluster_label] += 1. / distance

        for l in range(self.subgraph.n_clusters):

            # If the sum of internal and external clusters is bigger than 0
            if internal_cluster[l] + external_cluster[l] > 0.0:
                # Increments the value of the cut
                cut += external_cluster[l] / (internal_cluster[l] + external_cluster[l])

        return cut

    def _best_minimum_cut(self, min_k, max_k):
        """Performs a minimum cut on the subgraph using the best `k` value.

        Args:
            min_k (int): Minimum value of k.
            max_k (int): Maximum value of k.

        """

        logger.debug('Calculating the best minimum cut within [%d, %d] ...', min_k, max_k)

        # Calculates the maximum possible distances
        max_distances = self.subgraph.create_arcs(
            max_k, self.distance_fn, self.pre_computed_distance, self.pre_distances)

        # Initialize the minimum cut as maximum possible value
        min_cut = c.FLOAT_MAX

        best_k = self.max_k

        # for k in range(min_k, max_k + 1):
        for k in tqdm.trange(min_k, max_k + 1):

            # If minimum cut is different than zero
            if min_cut == 0.0:
                continue

            # Gathers the subgraph's density
            self.subgraph.density = max_distances[k - 1]

            # Gathers current `k` as the subgraph's best `k` value
            self.subgraph.best_k = k

            # Calculates the p.d.f.
            self.subgraph.calculate_pdf(k)

            # Clustering with current `k` value
            self._clustering(k)

            # Performs the normalized cut with current `k` value
            cut = self._normalized_cut(k)

            if cut < min_cut:
                min_cut = cut
                best_k = k

        self.subgraph.destroy_arcs()

        # Applying best `k` value to the subgraph
        self.subgraph.best_k = best_k

        # Creating new arcs with the best `k` value
        self.subgraph.create_arcs(best_k, self.distance_fn, self.pre_computed_distance, self.pre_distances)

        # Calculating the new p.d.f. with the best `k` value
        self.subgraph.calculate_pdf(best_k)

        logger.debug(f'Best: {best_k} | Minimum cut: {min_cut: .2f}.')

    def fit(self, X_train, Y_train=None, I_train=None):
        """Fits data in the classifier.

        Args:
            X_train (np.array): Array of training features.
            Y_train (np.array): Array of training labels.
            I_train (np.array): Array of training indexes.

        """

        logger.info('Clustering with classifier ...')

        start = time.time()

        # Creating a subgraph
        self.subgraph = NewKNNSubgraph(X_train, Y_train, I_train)

        # Performing the best minimum cut on the subgraph
        self._best_minimum_cut(self.min_k, self.max_k)

        # Clustering the data with best `k` value
        self._clustering(self.subgraph.best_k)

        # The subgraph has been properly trained
        self.subgraph.trained = True

        end = time.time()

        train_time = end - start

        logger.info('Classifier has been clustered with.')
        logger.info('Number of clusters: %d.', self.subgraph.n_clusters)
        logger.info('Clustering time: %s seconds.', train_time)

    def predict(self, X_val, I_val=None):
        """Predicts new data using the pre-trained classifier.

        Args:
            X_val (np.array): Array of validation features.
            I_val (np.array): Array of validation indexes.

        Returns:
            A list of predictions for each record of the data.

        """

        if not self.subgraph:
            raise e.BuildError('KNNSubgraph has not been properly created')

        if not self.subgraph.trained:
            raise e.BuildError('Classifier has not been properly clustered')

        logger.info('Predicting data ...')

        start = time.time()

        # Creating a prediction subgraph
        pred_subgraph = NewKNNSubgraph(X_val, I=I_val)

        # Gathering the best `k` value
        best_k = self.subgraph.best_k

        # for i in range(pred_subgraph.n_nodes):
        for i in tqdm.trange(pred_subgraph.n_nodes):

            i_neighbors, i_distances = [], []

            for j in range(self.subgraph.n_nodes):

                if self.pre_computed_distance:
                    dist = self.pre_distances[pred_subgraph.nodes[i].idx][self.subgraph.nodes[j].idx]
                else:
                    dist = self.distance_fn(pred_subgraph.nodes[i].features, self.subgraph.nodes[j].features)

                i_distances.insert(0, dist)
                i_neighbors.insert(0, j)

            # Sort nodes by distance and query only the top-k neighbors
            sorted_idx = np.argsort(i_distances)[:best_k]
            i_distances = np.array(i_distances)[sorted_idx].tolist()
            i_neighbors = np.array(i_neighbors)[sorted_idx].tolist()

            # Indices of neighbor nodes are added to node 'i' adjacency list
            pred_subgraph.nodes[i].adjacency = i_neighbors.copy()

            # Distances of neighbor nodes are added to node 'i' adjacency distance list
            pred_subgraph.nodes[i].adj_distances = i_distances.copy()

            # Array of k best distances (k-shared neighbors)
            distances = pred_subgraph.nodes[i].adj_distances[:best_k].copy()

            # Node density computed through a gaussian kernel using only adjacent nodes
            density = np.sum(np.exp(-np.array(distances) / self.subgraph.constant))

            # Gather its mean value
            density /= best_k

            # Scale the density between minimum and maximum values
            density = ((c.MAX_DENSITY - 1) * (density - self.subgraph.min_density) /
                       (self.subgraph.max_density - self.subgraph.min_density + c.EPSILON)) + 1

            neighbor_costs = [self.subgraph.nodes[neighbor].cost for neighbor in i_neighbors]

            # Calculate the temporary cost
            temp_cost = np.minimum(neighbor_costs, [density])

            # Select the maximum cost among node's neighbors
            k = np.argmax(temp_cost)

            # Gathers the node's neighbor
            neighbor = int(i_neighbors[k])

            pred_subgraph.nodes[i].density = density

            pred_subgraph.nodes[i].cost = self.subgraph.nodes[neighbor].cost

            # Propagates the predicted label from the neighbour
            pred_subgraph.nodes[i].predicted_label = self.subgraph.nodes[neighbor].predicted_label

            # Propagates the cluster label from the neighbour
            pred_subgraph.nodes[i].cluster_label = self.subgraph.nodes[neighbor].cluster_label

            # Creating the list of predictions
        preds = [pred.predicted_label for pred in pred_subgraph.nodes]

        # Creating the list of clusters
        clusters = [pred.cluster_label for pred in pred_subgraph.nodes]

        # Ending timer
        end = time.time()

        # Calculating prediction task time
        pred_time = end - start

        logger.info('Data has been predicted.')
        logger.info(f'Prediction time: {pred_time : .4f} seconds.')

        return preds, clusters, pred_subgraph

    def fit_predict(self, X_train, Y_train=None, I_train=None):

        self.fit(X_train, Y_train=Y_train, I_train=I_train)

        return np.array([node.cluster_label for node in self.subgraph.nodes])

    def propagate_labels(self):
        """Runs through the clusters and propagate the clusters roots labels to the samples.

        """

        logger.info('Assigning predicted labels from clusters ...')

        for i in range(self.subgraph.n_nodes):
            # Gathers the root from the node
            root = self.subgraph.nodes[i].root

            if root == i:
                self.subgraph.nodes[i].predicted_label = self.subgraph.nodes[i].label

            else:
                self.subgraph.nodes[i].predicted_label = self.subgraph.nodes[root].label

        logger.info('Labels assigned.')
