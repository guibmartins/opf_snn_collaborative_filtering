"""Unsupervised Optimum-Path Forest.
"""

import time
import numpy as np
import tqdm

import opfython.utils.constants as c
import opfython.math.distance as d
# import opfython.math.cf_distances as d
import opfython.utils.exception as e
import opfython.math.general as g
import opfython.utils.logging as log
from opfython.core import OPF
from opfython.core import Heap
from opfython.subgraphs.snn import SNNSubgraph
from opfython.algorithms import ann_algorithms
from opfython.math.distance import shared_near_neighbor_distance

logger = log.get_logger(__name__)
snn_distance = shared_near_neighbor_distance


class UnsupervisedSnnOPF(OPF):
    """An UnsupervisedOPF which implements the unsupervised version of OPF classifier.
    This implementation considers an adjacency relation based on Shared Near Neighbor (SNN) graph
    References:
        L. M. Rocha, F. A. M. Cappabianco, A. X. Falcão.
        Data clustering as an optimum-path forest problem with applications in image analysis.
        International Journal of Imaging Systems and Technology (2009).

    """

    def __init__(
            self, min_k=1, max_k=1, distance='squared_euclidean', pre_computed_distance=None, **kwargs):
        """Initialization method.

        Args:
            min_k (int): Minimum `k` value for cutting the subgraph.
            max_k (int): Maximum `k` value for cutting the subgraph.
            distance (str): An indicator of the distance metric to be used.
            pre_computed_distance (str): A pre-computed distance file for feeding into OPF.

        """

        # logger.info('Overriding class: OPF -> UnsupervisedOPF.')

        # Override its parent class with the receiving arguments
        super(UnsupervisedSnnOPF, self).__init__(distance, pre_computed_distance)

        # Defining the minimum `k` value for cutting the subgraph
        self.min_k = min_k

        # Defining the maximum `k` value for cutting the subgraph
        self.max_k = max_k

        self.fit_time = 0.0

        self.pred_time = 0.0

        if kwargs is not None:
            # Unpack additional function arguments
            self.remove_maxima = kwargs.get('remove_maxima')
            self.density_computation = kwargs.get('density_computation')

        # logger.info('Class overrided.')

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

    @property
    def remove_maxima(self):

        return self._remove_maxima

    @remove_maxima.setter
    def remove_maxima(self, remove_maxima):

        if remove_maxima is None:
            self._remove_maxima = {}

        else:
            if not isinstance(remove_maxima, dict):
                raise e.TypeError("`remove_maxima` should be a dictionary.")

            self._remove_maxima = remove_maxima

    @property
    def density_computation(self):

        return self._density_computation

    @density_computation.setter
    def density_computation(self, density_computation):

        if not isinstance(density_computation, str):
            raise e.TypeError('`density_computation` should be a valid string like `degree`, '
                              '`eigen_centrality`, and `pdf`')

        self._density_computation = density_computation

    def _clustering(self, n_neighbours):
        """Clusters the subgraph using a `k` value (number of neighbours).

        Args:
            n_neighbours (int): Number of neighbours to be used.

        """
        # For every possible node guarantee the symmetry of the adjacency relation
        for i in range(self.subgraph.n_nodes):

            for j in self.subgraph.nodes[i].shared_adjacency[:n_neighbours]:

                if i not in self.subgraph.nodes[j].shared_adjacency[:n_neighbours] and \
                        self.subgraph.nodes[j].density == self.subgraph.nodes[i].density:

                    dist = self.distance_fn(self.subgraph.nodes[i].features, self.subgraph.nodes[j].features)

                    self.subgraph.nodes[j].adjacency.insert(0, i)
                    self.subgraph.nodes[j].adj_distances.insert(0, dist)

                    dist = snn_distance(self.subgraph.nodes[i].adjacency[:n_neighbours],
                                        self.subgraph.nodes[j].adjacency[:n_neighbours])

                    self.subgraph.nodes[j].shared_adjacency.insert(0, i)
                    self.subgraph.nodes[j].shared_adj_distances.insert(0, dist)
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

        self.subgraph.idx_nodes.clear()

        # While the heap is not empty
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

            n_plateaus = self.subgraph.nodes[p].n_plateaus
            tmp_n_neighbors = len(self.subgraph.nodes[p].shared_adjacency[n_plateaus:n_plateaus + n_neighbours])

            # Calculates the number of its adjacent nodes
            n_adjacents = self.subgraph.nodes[p].n_plateaus + tmp_n_neighbors

            # For every possible adjacent node
            for k in range(n_adjacents):

                # Gathers the adjacent identifier
                q = int(self.subgraph.nodes[p].shared_adjacency[k])

                # If its color in the heap is different from `BLACK`
                if h.color[q] != c.BLACK:

                    # Calculates the current cost
                    current_cost = np.minimum(
                        h.cost[p], self.subgraph.nodes[q].density)

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

    def _delta(self, i, j):
        return 1 if self.subgraph.nodes[i].cluster_label == self.subgraph.nodes[j].cluster_label else 0

    def _modularity(self, n_neighbors):

        # Initialize the modularity variable
        modularity = 0.

        # Number of arcs in the graph
        m = 0

        A = np.zeros(shape=(self.subgraph.n_nodes, self.subgraph.n_nodes))

        # 1: Definir o número total de arestas do grafo (snn-graph)
        for i in range(self.subgraph.n_nodes):

            n_plateaus = int(self.subgraph.nodes[i].n_plateaus)
            tmp_n_neighbors = len(self.subgraph.nodes[i].shared_adjacency[n_plateaus:n_plateaus + n_neighbors])

            # Calculates its number of adjacent nodes
            n_adjacents = (n_plateaus + tmp_n_neighbors)
            # adj_i = np.unique(self.subgraph.nodes[i].shared_adjacency[:n_adjacents])
            adj_i = self.subgraph.nodes[i].shared_adjacency[:n_adjacents]

            if len(adj_i) > 0:
                A[i][adj_i] = 1
                A.T[i][adj_i] = 1

        # Grafo indireto: arcos simétricos
        m = 0.5 * np.sum(A)

        # 2: Calcular a modularidade de cada
        for i in range(self.subgraph.n_nodes):

            # n_plateaus = int(self.subgraph.nodes[i].n_plateaus)
            # tmp_n_neighbors = len(self.subgraph.nodes[i].shared_adjacency[n_plateaus:n_plateaus + n_neighbors])
            #
            # # Calculates its number of adjacent nodes
            # n_adjacents = (n_plateaus + tmp_n_neighbors)
            # n_adj_i = np.unique(self.subgraph.nodes[i].shared_adjacency[:n_adjacents]).shape[0]

            for j in range(self.subgraph.n_nodes):
                p_ij = A[i, :].sum() * A[j, :].sum() / (2. * m)
                modularity += (A[i][j] - p_ij) * self._delta(i, j)

        return modularity / (2. * m)

    def _modularity_maximization(self, min_k, max_k):
        """Performs a maximization of the graph modularity using the best `k` value.

                Args:
                    min_k (int): Minimum value of k.
                    max_k (int): Maximum value of k.

                """

        logger.debug('Calculating the best maximum modularity within [%d, %d] ...', min_k, max_k)

        # Calculates the maximum possible distances
        # max_distances = self.subgraph.build_arcs(max_k, self.ann_search)
        max_distances = self.subgraph.create_arcs(max_k, self.distance_fn, snn_distance)

        # Initialize the minimum cut as maximum possible value
        # min_cut = c.FLOAT_MAX
        max_q = -1 * c.FLOAT_MAX

        best_k = max_k

        # For every possible value of `k`
        for k in range(min_k, max_k + 1):

            # Gathers the subgraph's density
            self.subgraph.density = max_distances[k - 1]

            # Gathers current `k` as the subgraph's best `k` value
            self.subgraph.best_k = k

            # Calculates the p.d.f.
            self.subgraph.calculate_pdf(k)
            # self.subgraph.calculate_node_density(k)

            # Clustering with current `k` value
            self._clustering(k)

            # Performs the normalized cut with current `k` value
            # cut = self._normalized_cut(k)
            q = self._modularity(k)

            # print(f"Current computed modularity (k = {k}): {q:.4f}...")

            if k == 1:
                continue

            # If the cut's cost is smaller than minimum cut
            if q > max_q:
                # Replace its value
                max_q = q

                # And defines a new best `k` value
                best_k = k

        # Destroy current arcs
        self.subgraph.destroy_arcs()

        # Applying best `k` value to the subgraph
        self.subgraph.best_k = best_k

        # Creating new arcs with the best `k` value
        self.subgraph.create_arcs(best_k, self.distance_fn, snn_distance)

        # Calculating the new p.d.f. with the best `k` value
        self.subgraph.calculate_pdf(best_k)
        # self.subgraph.calculate_node_density(best_k)

        logger.debug(f'Best: {best_k} | Maximum modularity: {max_q: .4f}.')

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

        # For every possible node
        for i in range(self.subgraph.n_nodes):

            n_plateaus = self.subgraph.nodes[i].n_plateaus
            tmp_n_neighbors = len(self.subgraph.nodes[i].shared_adjacency[n_plateaus:n_plateaus + n_neighbours])

            # Calculates its number of adjacent nodes
            n_adjacents = self.subgraph.nodes[i].n_plateaus + tmp_n_neighbors
            # n_adjacents = self.subgraph.nodes[i].n_plateaus + n_neighbours

            distance = self.subgraph.nodes[i].shared_adj_distances[:n_adjacents]

            # For every possible adjacent node
            for k in range(n_adjacents):

                # Gathers its adjacent node identifier
                j = int(self.subgraph.nodes[i].shared_adjacency[k])

                # If distance is bigger than 0
                if distance[k] > 0:
                    # If nodes belongs to the same clusters
                    if self.subgraph.nodes[i].cluster_label == self.subgraph.nodes[j].cluster_label:
                        # Increments the internal cluster distance
                        internal_cluster[self.subgraph.nodes[i].cluster_label] += 1. / float(distance[k])

                    # If nodes belongs to distinct clusters
                    else:
                        # Increments the external cluster distance
                        external_cluster[self.subgraph.nodes[i].cluster_label] += 1. / float(distance[k])

        # For every possible cluster
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
        # max_distances = self.subgraph.build_arcs(max_k, self.ann_search)
        max_distances = self.subgraph.create_arcs(max_k, self.distance_fn, snn_distance)

        # Initialize the minimum cut as maximum possible value
        min_cut = c.FLOAT_MAX

        best_k = max_k

        # For every possible value of `k`
        # for k in range(min_k, max_k + 1):
        for k in tqdm.trange(min_k, max_k + 1):

            # If minimum cut is different than zero
            if min_cut == 0.0:
                continue

            # Gathers the subgraph's density
            self.subgraph.density = max_distances[k - 1]

            # Gathers current `k` as the subgraph's best `k` value
            self.subgraph.best_k = k

            # Calculates nodes weights
            self.subgraph.density_fn(self.subgraph, k)

            # Clustering with current `k` value
            self._clustering(k)

            # Performs the normalized cut with current `k` value
            cut = self._normalized_cut(k)

            # If the cut's cost is smaller than minimum cut
            if cut < min_cut:

                # Replace its value
                min_cut = cut

                # And defines a new best `k` value
                best_k = k

        # Destroy current arcs
        self.subgraph.destroy_arcs()

        # Applying best `k` value to the subgraph
        self.subgraph.best_k = best_k

        # Creating new arcs with the best `k` value
        self.subgraph.create_arcs(best_k, self.distance_fn, snn_distance)

        # Calculates nodes weights (density) with the best `k` value
        self.subgraph.density_fn(self.subgraph, best_k)

        logger.debug(f'Best: {best_k} | Minimum cut: {min_cut: .2f}.')

    def fit(self, X_train, Y_train=None, I_train=None):
        """Fits data in the classifier.

        Args:
            X_train (np.array): Array of training features.
            Y_train (np.array): Array of training labels.
            I_train (np.array): Array of training indexes.

        """

        logger.info('Clustering with classifier ...')

        # Initializing the timer
        start = time.time()

        # Creating a subgraph
        self.subgraph = SNNSubgraph(
            X_train, Y=Y_train, I=I_train, density_fn=self.density_computation)

        # Performing the best minimum cut on the subgraph
        self._best_minimum_cut(self.min_k, self.max_k)

        if len(self.remove_maxima) > 0:

            if list(self.remove_maxima.keys())[0] \
                    in ['height', 'area', 'volume']:

                key, value = self.remove_maxima.popitem()

                if key == 'height':
                    self.subgraph.eliminate_maxima_height(value)
                elif key == 'area':
                    self.subgraph.eliminate_maxima_area(value)
                else:
                    self.subgraph.eliminate_maxima_volume(value)

        # Clustering the data with best `k` value
        self._clustering(self.subgraph.best_k)

        # The subgraph has been properly trained
        self.subgraph.trained = True

        # Ending timer
        end = time.time()

        # Calculating training task time
        self.fit_time = end - start

        logger.info('Classifier has been clustered with.')
        logger.info(f'Number of clusters: {self.subgraph.n_clusters}.')
        logger.info(f'Clustering time: {self.fit_time : .4f} seconds.')

    def predict(self, X_val, I_val=None):

        if self.density_computation == 'pdf':
            return self.predict_shared(X_val, I_val)

        elif self.density_computation == 'eigen_centrality':
            return self.predict_shared_centrality(X_val, I_val)

        else:
            return self.predict_alternative(X_val, I_val)

    def predict_shared(self, X_val, I_val=None):
        """Predicts new data using the pre-trained classifier.

        Args:
            X_val (np.array): Array of validation features.
            I_val (np.array): Array of validation indexes.

        Returns:
            A list of predictions for each record of the data.

        """

        # Checks if there is a knn-subgraph
        if not self.subgraph:
            # If not, raises an BuildError
            raise e.BuildError('SNNSubgraph has not been properly created')

        # Checks if knn-subgraph has been properly trained
        if not self.subgraph.trained:
            # If not, raises an BuildError
            raise e.BuildError('Classifier has not been properly clustered')

        logger.info('Predicting data ...')

        # Initializing the timer
        start = time.time()

        # Creating a prediction subgraph
        pred_subgraph = SNNSubgraph(X_val, I=I_val)

        # Gathering the best `k` value
        best_k = self.subgraph.best_k

        # For every possible prediction node
        for i in tqdm.trange(pred_subgraph.n_nodes):

            # For every possible trained node
            i_neighbors, i_distances = [], []

            # For every node (training set)
            for j in range(self.subgraph.n_nodes):

                dist = self.distance_fn(
                    pred_subgraph.nodes[i].features, self.subgraph.nodes[j].features)

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

            for j, j_dist in zip(i_neighbors, i_distances):

                j_adjacent = self.subgraph.nodes[j].adjacency[:best_k].copy()
                j_distances = self.subgraph.nodes[j].adj_distances[:best_k].copy()

                idx = np.searchsorted(j_distances, j_dist)
                j_adjacent.insert(idx, i)
                j_distances.insert(idx, j_dist)

                dist = 1.
                if i in j_adjacent[:best_k]:
                    dist = d.shared_near_neighbor_distance(i_neighbors[:best_k], j_adjacent[:best_k])

                sorted_idx = np.searchsorted(pred_subgraph.nodes[i].shared_adj_distances, dist)
                pred_subgraph.nodes[i].shared_adjacency.insert(sorted_idx, j)
                pred_subgraph.nodes[i].shared_adj_distances.insert(sorted_idx, dist)

            # Array of k best distances (k-shared neighbors)
            distances = pred_subgraph.nodes[i].shared_adj_distances[:best_k].copy()

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
        self.pred_time = end - start

        logger.info('Data has been predicted.')
        logger.info(f'Prediction time: {self.pred_time : .4f} seconds.')

        return preds, clusters, pred_subgraph

    def predict_shared_centrality(self, X_val, I_val=None):
        """Predicts new data using the pre-trained classifier.

                Args:
                    X_val (np.array): Array of validation features.
                    I_val (np.array): Array of validation indexes.

                Returns:
                    A list of predictions for each record of the data.

                """

        # Checks if there is a knn-subgraph
        if not self.subgraph:
            # If not, raises an BuildError
            raise e.BuildError('SNNSubgraph has not been properly created')

        # Checks if knn-subgraph has been properly trained
        if not self.subgraph.trained:
            # If not, raises an BuildError
            raise e.BuildError('Classifier has not been properly clustered')

        logger.info('Predicting data ...')

        # Initializing the timer
        start = time.time()

        # Creating a prediction subgraph
        pred_subgraph = SNNSubgraph(
            X_val, I=I_val, density_fn=self.density_computation)

        # Gathering the best `k` value
        best_k = self.subgraph.best_k

        # For every possible prediction node
        for i in tqdm.trange(pred_subgraph.n_nodes):

            # For every possible trained node
            i_neighbors, i_distances = [], []

            # For every node (training set)
            for j in range(self.subgraph.n_nodes):

                dist = self.distance_fn(
                    pred_subgraph.nodes[i].features, self.subgraph.nodes[j].features)

                i_distances.insert(0, dist)
                i_neighbors.insert(0, j)

            # Sort by distance and query only the top-k neighbors
            sorted_idx = np.argsort(i_distances)[:best_k]
            i_distances = np.array(i_distances)[sorted_idx].tolist()
            i_neighbors = np.array(i_neighbors)[sorted_idx].tolist()

            # Indices of neighbor nodes are added to node 'i' adjacency list
            pred_subgraph.nodes[i].adjacency = i_neighbors.copy()

            # Distances of neighbor nodes are added to no 'i' adjacency distance list
            pred_subgraph.nodes[i].adj_distances = i_distances.copy()

            for j, j_dist in zip(i_neighbors, i_distances):

                j_adjacent = self.subgraph.nodes[j].adjacency[:best_k].copy()
                j_distances = self.subgraph.nodes[j].adj_distances[:best_k].copy()

                idx = np.searchsorted(j_distances, j_dist)
                j_adjacent.insert(idx, i)
                j_distances.insert(idx, j_dist)

                dist = 1.
                if i in j_adjacent[:best_k]:
                    dist = d.shared_near_neighbor_distance(i_neighbors[:best_k], j_adjacent[:best_k])

                sorted_idx = np.searchsorted(pred_subgraph.nodes[i].shared_adj_distances, dist)
                pred_subgraph.nodes[i].shared_adjacency.insert(sorted_idx, j)
                pred_subgraph.nodes[i].shared_adj_distances.insert(sorted_idx, dist)

            # Array of k-shared adjacent nodes
            adjacents = pred_subgraph.nodes[i].shared_adjacency[:best_k]

            # Array of k best distances (k-shared neighbors)
            # distances = pred_subgraph.nodes[i].shared_adj_distances[:best_k]

            nn_list = adjacents.copy()
            nn_list.insert(0, i)

            id_map = {adj: j for j, adj in enumerate(nn_list)}

            # Calculate the adjacency relation restricted to the nearest neighbors
            W = np.zeros((len(nn_list), len(nn_list)), dtype=np.float64)

            for idx in id_map:
                if idx == 0:
                    adjs = pred_subgraph.nodes[idx].shared_adjacency[:best_k]
                    dists = pred_subgraph.nodes[idx].shared_adj_distances[:best_k]
                else:
                    adjs = self.subgraph.nodes[idx].shared_adjacency[:best_k]
                    dists = self.subgraph.nodes[idx].shared_adj_distances[:best_k]

                nn, nn_idx, _ = np.intersect1d(adjs, list(id_map.keys()), return_indices=True)
                indices = [id_map[idx] for idx in nn]
                # W[id_map[idx], indices] = 1. - np.array(dists)[nn_idx]
                W[id_map[idx], indices] = W.T[id_map[idx], indices] = 1. - np.array(dists)[nn_idx]

            # Calculate the adjacency relation restricted to its neighbors
            # W = np.zeros((len(adjacents) + 1, len(adjacents) + 1), dtype=np.float64)
            # W[0, 1:] = W[1:, 0] = 1. - np.asfarray(distances, dtype=np.float64)

            # Normalized eigen decomposition of A: The eigenvector
            # corresponding to the maximum (principal) eigenvalue is chosen
            eigvec, eigval = g.norm_eigen_centrality(W)

            # Principal eigen value acts as a normalization factor
            density = np.abs(eigvec[0]) / (eigval + c.EPSILON)

            # Scale the density between minimum and maximum values
            density = ((c.MAX_DENSITY - 1.) * (density - self.subgraph.min_density) /
                       (self.subgraph.max_density - self.subgraph.min_density + c.EPSILON)) + 1.

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
        self.pred_time = end - start

        logger.info('Data has been predicted.')
        logger.info(f'Prediction time: {self.pred_time : .4f} seconds.')

        return preds, clusters, pred_subgraph

    def predict_alternative(self, X_val, I_val=None):
        """Predicts new data using the pre-trained classifier.

        Args:
            X_val (np.array): Array of validation features.
            I_val (np.array): Array of validation indexes.

        Returns:
            A list of predictions for each record of the data.

        """

        # Checks if there is a knn-subgraph
        if not self.subgraph:
            # If not, raises an BuildError
            raise e.BuildError('SNNSubgraph has not been properly created')

        # Checks if knn-subgraph has been properly trained
        if not self.subgraph.trained:
            # If not, raises an BuildError
            raise e.BuildError('Classifier has not been properly clustered')

        logger.info('Predicting data ...')

        # Initializing the timer
        start = time.time()

        # Creating a prediction subgraph
        pred_subgraph = SNNSubgraph(X_val, I=I_val)

        # Gathering the best `k` value
        best_k = self.subgraph.best_k

        # For every possible prediction node
        for i in tqdm.trange(pred_subgraph.n_nodes):

            # For every possible trained node
            i_neighbors, i_distances = [], []

            # For every node (training set)
            for j in range(self.subgraph.n_nodes):

                dist = self.distance_fn(
                    pred_subgraph.nodes[i].features,
                    self.subgraph.nodes[j].features)

                i_distances.insert(0, dist)
                i_neighbors.insert(0, j)

            # Sort by distance and query only the top-k neighbors
            sorted_idx = np.argsort(i_distances)[:best_k]

            i_distances = np.array(i_distances)[sorted_idx].tolist()
            i_neighbors = np.array(i_neighbors)[sorted_idx].tolist()

            # Indices of neighbor nodes are added to node 'i' adjacency list
            pred_subgraph.nodes[i].adjacency = i_neighbors

            # Distances of neighbor nodes are added to no 'i' adjacency distance list
            pred_subgraph.nodes[i].adj_distances = i_distances

            # Compute the shared adjacency of the node 'i'
            for j, j_dist in zip(i_neighbors, i_distances):

                j_adjacent = self.subgraph.nodes[j].adjacency[:best_k].copy()
                j_distances = self.subgraph.nodes[j].adj_distances[:best_k].copy()

                # Sorted insertion of the test node as adjacent of its neighbor
                idx = np.searchsorted(j_distances, j_dist)
                j_adjacent.insert(idx, i)
                j_distances.insert(idx, j_dist)

                dist = 1.
                if i in j_adjacent[:best_k]:
                    dist = d.shared_near_neighbor_distance(i_neighbors[:best_k], j_adjacent[:best_k])

                sorted_idx = np.searchsorted(pred_subgraph.nodes[i].shared_adj_distances, dist)
                pred_subgraph.nodes[i].shared_adjacency.insert(sorted_idx, j)
                pred_subgraph.nodes[i].shared_adj_distances.insert(sorted_idx, dist)

            # Fast cluster/label propagation proposed by Cappabianco et al. (2012) in
            # 'Brain tissue MR-image segmentation via optimum-path forest clustering (2012)'
            for j in self.subgraph.idx_nodes:

                dist = d.shared_near_neighbor_distance(
                    pred_subgraph.nodes[i].shared_adjacency[:best_k],
                    self.subgraph.nodes[j].shared_adjacency[:best_k])

                if dist <= self.subgraph.nodes[j].radius:

                    # Neighbour becomes the predecessor of test node i
                    pred_subgraph.nodes[i].pred = j

                    # Propagates the root identifier from the predecessor
                    pred_subgraph.nodes[i].root = self.subgraph.nodes[j].root

                    # Propagates the path-cost from the neighbour
                    pred_subgraph.nodes[i].cost = self.subgraph.nodes[j].cost

                    # Propagates the density from the neighbour
                    pred_subgraph.nodes[i].density = self.subgraph.nodes[j].density

                    # Propagates the predicted label from the neighbour
                    pred_subgraph.nodes[i].predicted_label = self.subgraph.nodes[j].predicted_label

                    # Propagates the cluster label from the neighbour
                    pred_subgraph.nodes[i].cluster_label = self.subgraph.nodes[j].cluster_label

                    break

        # Creating the list of predictions
        preds = [pred.predicted_label for pred in pred_subgraph.nodes]

        # Creating the list of clusters
        clusters = [pred.cluster_label for pred in pred_subgraph.nodes]

        # Ending timer
        end = time.time()

        # Calculating prediction task time
        self.pred_time = end - start

        logger.info('Data has been predicted.')
        logger.info(f'Prediction time: {self.pred_time : .4f} seconds.')

        return preds, clusters, pred_subgraph

    def fit_predict(self, X_train, Y_train=None, I_train=None):

        self.fit(X_train, Y_train=Y_train, I_train=I_train)

        # Creating the list of clusters
        clusters = [node.cluster_label for node in self.subgraph.nodes]

        # Creating the list of predicted labels
        preds = [node.predicted_label for node in self.subgraph.nodes]

        return preds, clusters

    def propagate_labels(self):
        """Runs through the clusters and propagate the clusters roots labels to the samples.

        """

        logger.info('Assigning predicted labels from clusters ...')

        # For every possible node
        for i in range(self.subgraph.n_nodes):
            # Gathers the root from the node
            root = self.subgraph.nodes[i].root

            # If the root is the same as node's identifier
            if root == i:
                # Apply the predicted label as node's label
                self.subgraph.nodes[i].predicted_label = self.subgraph.nodes[i].label

            # If the root is different from node's identifier
            else:
                # Apply the predicted label as the root's label
                self.subgraph.nodes[i].predicted_label = self.subgraph.nodes[root].label

        logger.info('Labels assigned.')


class UnsupervisedAnnOPF(OPF):
    """An UnsupervisedOPF which implements the unsupervised version of OPF classifier.
    This implementation considers an adjacency relation based on Shared Near Neighbor (SNN) graph
    References:
        L. M. Rocha, F. A. M. Cappabianco, A. X. Falcão.
        Data clustering as an optimum-path forest problem with applications in image analysis.
        International Journal of Imaging Systems and Technology (2009).

    """

    def __init__(
            self, min_k=1, max_k=1, distance='euclidean', eliminate_maxima=None,
            pre_computed_distance=None, ann_params=None):
        """Initialization method.

        Args:
            min_k (int): Minimum `k` value for cutting the subgraph.
            max_k (int): Maximum `k` value for cutting the subgraph.
            distance (str): An indicator of the distance metric to be used.
            pre_computed_distance (str): A pre-computed distance file for feeding into OPF.

        """

        logger.info('Overriding class: OPF -> UnsupervisedOPF.')

        # Override its parent class with the receiving arguments
        super(UnsupervisedAnnOPF, self).__init__(distance, pre_computed_distance)

        # Defining the minimum `k` value for cutting the subgraph
        self.min_k = min_k

        # Defining the maximum `k` value for cutting the subgraph
        self.max_k = max_k

        self.ann_params = ann_params

        self.ann_search = None

        self.fit_time = 0.

        self.pred_time = 0.

        self.eliminate_maxima = {} if eliminate_maxima is None else eliminate_maxima

        # Defining the ann search method
        if ann_params.get('name') == 'annoy':
            self.ann_class = ann_algorithms.Annoy

        elif ann_params.get('name') == 'hnsw':
            self.ann_class = ann_algorithms.HNSW

        elif ann_params.get('name') == 'kdtree':
            self.ann_class = ann_algorithms.KD_Tree

        else:
            self.ann_class = ann_algorithms.NNeighbors

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
        # For every possible node guarantee the symmetry of the adjacency relation
        for i in range(self.subgraph.n_nodes):

            for j in self.subgraph.nodes[i].shared_adjacency[:n_neighbours]:

                if i not in self.subgraph.nodes[j].shared_adjacency[:n_neighbours] and \
                        self.subgraph.nodes[j].density == self.subgraph.nodes[i].density:
                    dist = self.distance_fn(self.subgraph.nodes[i].features, self.subgraph.nodes[j].features)

                    self.subgraph.nodes[j].adjacency.insert(0, i)
                    self.subgraph.nodes[j].adj_distances.insert(0, dist)

                    dist = snn_distance(self.subgraph.nodes[i].adjacency[:n_neighbours],
                                        self.subgraph.nodes[j].adjacency[:n_neighbours])

                    self.subgraph.nodes[j].shared_adjacency.insert(0, i)
                    self.subgraph.nodes[j].shared_adj_distances.insert(0, dist)
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

        self.subgraph.idx_nodes.clear()

        # While the heap is not empty
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

            n_plateaus = self.subgraph.nodes[p].n_plateaus
            tmp_n_neighbors = len(self.subgraph.nodes[p].shared_adjacency[n_plateaus:n_plateaus + n_neighbours])

            # Calculates the number of its adjacent nodes
            n_adjacents = self.subgraph.nodes[p].n_plateaus + tmp_n_neighbors

            # For every possible adjacent node
            for k in range(n_adjacents):

                # Gathers the adjacent identifier
                q = int(self.subgraph.nodes[p].shared_adjacency[k])

                # If its color in the heap is different from `BLACK`
                if h.color[q] != c.BLACK:

                    # Calculates the current cost
                    current_cost = np.minimum(
                        h.cost[p], self.subgraph.nodes[q].density)

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

        # For every possible node
        for i in range(self.subgraph.n_nodes):

            n_plateaus = self.subgraph.nodes[i].n_plateaus
            tmp_n_neighbors = len(self.subgraph.nodes[i].shared_adjacency[n_plateaus:n_plateaus + n_neighbours])

            # Calculates its number of adjacent nodes
            n_adjacents = self.subgraph.nodes[i].n_plateaus + tmp_n_neighbors
            # n_adjacents = self.subgraph.nodes[i].n_plateaus + n_neighbours

            distance = self.subgraph.nodes[i].shared_adj_distances[:n_adjacents]

            # For every possible adjacent node
            for k in range(n_adjacents):

                # Gathers its adjacent node identifier
                j = int(self.subgraph.nodes[i].shared_adjacency[k])

                # If distance is bigger than 0
                if distance[k] > 0:
                    # If nodes belongs to the same clusters
                    if self.subgraph.nodes[i].cluster_label == self.subgraph.nodes[j].cluster_label:
                        # Increments the internal cluster distance
                        internal_cluster[self.subgraph.nodes[i].cluster_label] += 1. / float(distance[k])

                    # If nodes belongs to distinct clusters
                    else:
                        # Increments the external cluster distance
                        external_cluster[self.subgraph.nodes[i].cluster_label] += 1. / float(distance[k])

        # For every possible cluster
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
        # max_distances = self.subgraph.build_arcs(max_k, self.ann_search)
        max_distances = self.subgraph.create_ann_arcs(max_k, snn_distance, self.ann_search)

        # Initialize the minimum cut as maximum possible value
        min_cut = c.FLOAT_MAX

        best_k = max_k

        # For every possible value of `k`
        for k in range(min_k, max_k + 1):

            if min_cut == 0.0:
                continue

            # If minimum cut is different than zero
            # if min_cut != 0.0:

            # Gathers the subgraph's density
            self.subgraph.density = max_distances[k - 1]

            # Gathers current `k` as the subgraph's best `k` value
            self.subgraph.best_k = k

            # Calculates the p.d.f.
            self.subgraph.calculate_pdf(k)
            # self.subgraph.calculate_node_density(k)

            # Clustering with current `k` value
            self._clustering(k)

            # Performs the normalized cut with current `k` value
            cut = self._normalized_cut(k)

            # If the cut's cost is smaller than minimum cut
            if cut < min_cut:
                # Replace its value
                min_cut = cut

                # And defines a new best `k` value
                best_k = k

        # Destroy current arcs
        self.subgraph.destroy_arcs()

        # Applying best `k` value to the subgraph
        self.subgraph.best_k = best_k

        # Creating new arcs with the best `k` value
        self.subgraph.create_ann_arcs(best_k, snn_distance, self.ann_search)

        # Calculating the new p.d.f. with the best `k` value
        self.subgraph.calculate_pdf(best_k)
        # self.subgraph.calculate_node_density(best_k)

        logger.debug(f'Best: {best_k} | Minimum cut: {min_cut: .4f}.')

    def fit(self, X_train, Y_train=None, I_train=None):
        """Fits data in the classifier.

        Args:
            X_train (np.array): Array of training features.
            Y_train (np.array): Array of training labels.
            I_train (np.array): Array of training indexes.

        """

        logger.info('Clustering with classifier ...')

        # Initializing the timer
        start = time.time()

        # Creating a subgraph
        self.subgraph = SNNSubgraph(X_train, Y_train, I_train)

        # Initiating the ANN method to perform Approximate Nearest Neighbors search
        if self.ann_params.get('name') == 'hnsw':
            self.ann_params['n_samples'] = X_train.shape[0]
            self.ann_params['n_features'] = X_train.shape[1]
            self.ann_params['ef'] = self.max_k

        if self.ann_params.get('name') == 'nneighbors':
            self.ann_params['n_neighbors'] = self.max_k

        self.ann_search = self.ann_class(self.ann_params)

        # Build ANN index structure
        self.ann_search.fit(X_train)

        # Performing the best minimum cut on the subgraph
        self._best_minimum_cut(self.min_k, self.max_k)

        if len(self.eliminate_maxima) > 0:

            if list(self.eliminate_maxima.keys())[0] in ['height', 'area', 'volume']:

                key, value = self.eliminate_maxima.popitem()

                if key == 'height':
                    self.subgraph.eliminate_maxima_height(value)
                elif key == 'area':
                    self.subgraph.eliminate_maxima_area(value)
                else:
                    self.subgraph.eliminate_maxima_volume(value)

        # Clustering the data with best `k` value
        self._clustering(self.subgraph.best_k)

        # The subgraph has been properly trained
        self.subgraph.trained = True

        # Ending timer
        end = time.time()

        # Calculating training task time
        self.fit_time = end - start

        logger.info('Classifier has been clustered with.')
        logger.info(f'Number of clusters: {self.subgraph.n_clusters}.')
        logger.info(f'Clustering time: {self.fit_time : .4f} seconds.')

    def predict(self, X_val, I_val=None):
        """Predicts new data using the pre-trained classifier.

        Args:
            X_val (np.array): Array of validation features.
            I_val (np.array): Array of validation indexes.

        Returns:
            A list of predictions for each record of the data.

        """

        # Checks if there is a knn-subgraph
        if not self.subgraph:
            # If not, raises an BuildError
            raise e.BuildError('ANNSubgraph has not been properly created')

        # Checks if knn-subgraph has been properly trained
        if not self.subgraph.trained:
            # If not, raises an BuildError
            raise e.BuildError('Classifier has not been properly clustered')

        logger.info('Predicting data ...')

        # Initializing the timer
        start = time.time()

        # Creating a prediction subgraph
        pred_subgraph = SNNSubgraph(X_val, I=I_val)

        # Gathering the best `k` value
        best_k = self.subgraph.best_k

        # For every possible prediction node
        for i in range(pred_subgraph.n_nodes):

            # For every possible trained node
            neighbors_idx, distances = self.ann_search.query(pred_subgraph.nodes[i].features, best_k)

            density = np.sum(np.exp(-np.array(distances) / self.subgraph.constant))

            # Gather its mean value
            density /= best_k

            # Scale the density between minimum and maximum values
            density = ((c.MAX_DENSITY - 1) * (density - self.subgraph.min_density) /
                       (self.subgraph.max_density - self.subgraph.min_density + c.EPSILON)) + 1

            neighbor_costs = [self.subgraph.nodes[neighbor].cost for neighbor in neighbors_idx]

            # Calculate the temporary cost
            temp_cost = np.minimum(neighbor_costs, [density])

            # Select the maximum cost among node's neighbors
            k = np.argmax(temp_cost)

            # Gathers the node's neighbor
            neighbor = int(neighbors_idx[k])

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
        self.pred_time = end - start

        logger.info('Data has been predicted.')
        logger.info(f'Prediction time: {self.pred_time : .4f} seconds.')

        return preds, clusters

    def fit_predict(self, X_train, Y_train=None, I_train=None):

        self.fit(X_train, Y_train, I_train)

        preds = [node.predicted_label for node in self.subgraph.nodes]

        clusters = [node.cluster_label for node in self.subgraph.nodes]

        return preds, clusters

    def predict_shared(self, X_val, I_val=None):
        """Predicts new data using the pre-trained classifier.

         Args:
             X_val (np.array): Array of validation features.
             I_val (np.array): Array of validation indexes.

         Returns:
             A list of predictions for each record of the data.

         """

        # Checks if there is a knn-subgraph
        if not self.subgraph:
            # If not, raises an BuildError
            raise e.BuildError('SNNSubgraph has not been properly created')

        # Checks if knn-subgraph has been properly trained
        if not self.subgraph.trained:
            # If not, raises an BuildError
            raise e.BuildError('Classifier has not been properly clustered')

        logger.info('Predicting data ...')

        # Initializing the timer
        start = time.time()

        # Creating a prediction subgraph
        pred_subgraph = SNNSubgraph(X_val, I=I_val)

        # Gathering the best `k` value
        best_k = self.subgraph.best_k

        # For every possible prediction node
        for i in range(pred_subgraph.n_nodes):

            # For every possible trained node
            i_neighbors, i_distances = self.ann_search.query(pred_subgraph.nodes[i].features, best_k)

            # Indices of neighbor nodes are added to node 'i' adjacency list
            pred_subgraph.nodes[i].adjacency = i_neighbors

            # Distances of neighbor nodes are added to no 'i' adjacency distance list
            pred_subgraph.nodes[i].adj_distances = i_distances

            for j, j_dist in zip(i_neighbors, i_distances):

                j_adjacent = self.subgraph.nodes[j].adjacency[:best_k].copy()
                j_distances = self.subgraph.nodes[j].adj_distances[:best_k].copy()

                idx = np.searchsorted(j_distances, j_dist)
                j_adjacent.insert(idx, i)
                j_distances.insert(idx, j_dist)

                dist = 1.
                if i in j_adjacent[:best_k]:
                    dist = d.shared_near_neighbor_distance(i_neighbors[:best_k], j_adjacent[:best_k])

                sorted_idx = np.searchsorted(pred_subgraph.nodes[i].shared_adj_distances, dist)
                pred_subgraph.nodes[i].shared_adjacency.insert(sorted_idx, j)
                pred_subgraph.nodes[i].shared_adj_distances.insert(sorted_idx, dist)

            # Array of k best distances (k-shared neighbors)
            distances = pred_subgraph.nodes[i].shared_adj_distances[:best_k]

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
        self.pred_time = end - start

        logger.info('Data has been predicted.')
        logger.info(f'Prediction time: {self.pred_time : .4f} seconds.')

        return preds, clusters, pred_subgraph

    def propagate_labels(self):
        """Runs through the clusters and propagate the clusters roots labels to the samples.

        """

        logger.info('Assigning predicted labels from clusters ...')

        # For every possible node
        for i in range(self.subgraph.n_nodes):
            # Gathers the root from the node
            root = self.subgraph.nodes[i].root

            # If the root is the same as node's identifier
            if root == i:
                # Apply the predicted label as node's label
                self.subgraph.nodes[i].predicted_label = self.subgraph.nodes[i].label

            # If the root is different from node's identifier
            else:
                # Apply the predicted label as the root's label
                self.subgraph.nodes[i].predicted_label = self.subgraph.nodes[root].label

        logger.info('Labels assigned.')


class HierarchicalUnsupervisedOPF(UnsupervisedSnnOPF):
    """An agglomerative-based hierarchical OPF clustering which
    implements the SNN-based unsupervised version of OPF classifier.

    """

    def __init__(
            self, min_k=1, max_k=1, distance='euclidean', pre_computed_distance=None, **kwargs):
        """Initialization method.

        Args:
            min_k (int): Minimum `k` value for cutting the subgraph.
            max_k (int): Maximum `k` value for cutting the subgraph.
            distance (str): An indicator of the distance metric to be used.
            pre_computed_distance (str): A pre-computed distance file for feeding into OPF.

        """
        # Override its parent class with the receiving arguments
        super(HierarchicalUnsupervisedOPF, self).__init__(
            min_k, max_k, distance, pre_computed_distance, **kwargs)

        self.sg_hierarchy = []

        self.clusters_hierarchy = []

        self.propagate_true_labels = False

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

        if Y_train is None:
            self.propagate_true_labels = False

        while n_clusters > 1:
            print('Clustering on layer', layer, '...')

            opf = UnsupervisedSnnOPF(max_k=max_k, distance=self.distance,
                                     pre_computed_distance=self.pre_computed_distance)

            preds, clusters = opf.fit_predict(X_layer, Y_train, I_train)

            n_clusters = opf.subgraph.n_clusters

            # Stack prototypes to compose the training set with respect to layer l
            X_layer = np.asfarray([i.features
                                   for label in range(n_clusters)
                                   for i in opf.subgraph.nodes
                                   if (i.pred == -1 and i.cluster_label == label)])

            self.sg_hierarchy.append(opf)

            if self.propagate_true_labels is True:

                opf.propagate_labels()
                self.clusters_hierarchy.append(np.array(preds))

            else:

                self.clusters_hierarchy.append(np.array(clusters))

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
