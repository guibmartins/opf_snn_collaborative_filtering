"""SNN-based Subgraph.
"""

import numpy as np

import opfython.utils.constants as c
import opfython.utils.exception as e
import opfython.math.general as g
import opfython.utils.logging as log
from opfython.core import Subgraph

logger = log.get_logger(__name__)


class SNNSubgraph(Subgraph):
    """An ANNSubgraph is used to implement an approximate nearest neighbours subgraph.

    """

    def __init__(self, X=None, Y=None, I=None, from_file=None, density_fn='pdf'):
        """Initialization method.

        Args:
            X (np.array): Array of features.
            Y (np.array): Array of labels.
            I (np.array): Array of indexes.
            from_file (bool): Whether Subgraph should be directly created from a file.

        """

        # Override its parent class with the receiving arguments
        super(SNNSubgraph, self).__init__(X, Y, I, from_file)

        #  Number of assigned clusters
        self.n_clusters = 0

        # Number of adjacent nodes (k-nearest neighbours)
        self.best_k = 0

        # Constant used to calculate the p.d.f.
        self.constant = 0.0

        # Density of the subgraph
        self.density = 0.0

        # Minimum density of the subgraph
        self.min_density = 0.0

        # Maximum density of the subgraph
        self.max_density = 0.0

        # Function to compute node density
        self.density_fn = self.DENSITIES[density_fn]

    @property
    def n_clusters(self):
        """int: Number of assigned clusters.

        """

        return self._n_clusters

    @n_clusters.setter
    def n_clusters(self, n_clusters):
        if not isinstance(n_clusters, int):
            raise e.TypeError('`n_clusters` should be an integer')
        if n_clusters < 0:
            raise e.ValueError('`n_clusters` should be >= 0')

        self._n_clusters = n_clusters

    @property
    def best_k(self):
        """int: Number of adjacent nodes (k-nearest neighbours).

        """

        return self._best_k

    @best_k.setter
    def best_k(self, best_k):
        if not isinstance(best_k, int):
            raise e.TypeError('`best_k` should be an integer')
        if best_k < 0:
            raise e.ValueError('`best_k` should be >= 0')

        self._best_k = best_k

    @property
    def constant(self):
        """float: Constant used to calculate the probability density function.

        """

        return self._constant

    @constant.setter
    def constant(self, constant):
        if not isinstance(constant, (float, int, np.int32, np.int64)):
            raise e.TypeError('`constant` should be a float or integer')

        self._constant = constant

    @property
    def density(self):
        """float: Density of the subgraph.

        """

        return self._density

    @density.setter
    def density(self, density):
        if not isinstance(density, (float, int, np.int32, np.int64)):
            raise e.TypeError('`density` should be a float or integer')

        self._density = density

    @property
    def min_density(self):
        """float: Minimum density of the subgraph.

        """

        return self._min_density

    @min_density.setter
    def min_density(self, min_density):
        if not isinstance(min_density, (float, int, np.int32, np.int64)):
            raise e.TypeError('`min_density` should be a float or integer')

        self._min_density = min_density

    @property
    def max_density(self):
        """float: Maximum density of the subgraph.

        """

        return self._max_density

    @max_density.setter
    def max_density(self, max_density):
        if not isinstance(max_density, (float, int, np.int32, np.int64)):
            raise e.TypeError('`max_density` should be a float or integer')

        self._max_density = max_density

    @property
    def density_fn(self):
        """float: Density of the subgraph.

        """

        return self._density_fn

    @density_fn.setter
    def density_fn(self, density_fn):
        if not callable(density_fn):
            raise e.TypeError('`density_fn` should be a callable')

        self._density_fn = density_fn

    def rescale_densities(self, densities):

        # Rescaling densities within the interval [min_density, max_density]
        if self.min_density == self.max_density:
            densities[:] = c.MAX_DENSITY + 0.
        else:
            norm_dens = (densities - self.min_density) / \
                        (self.max_density - self.min_density + c.EPSILON)
            densities[:] = (c.MAX_DENSITY - 1.) * norm_dens + 1.

        for i in range(self.n_nodes):

            # Calculates the node's density
            self.nodes[i].density = densities[i]

            # Calculates the node's cost
            self.nodes[i].cost = densities[i] - 1.

    def calculate_pdf(self, n_neighbours):
        """Calculates the probability density function for `k` neighbours.

        Args:
            n_neighbours (int): Number of neighbours in the adjacency relation.

        """

        # Calculating constant for computing the probability density function
        self.constant = 2 * self.density / 9.

        # Defining subgraph's minimum density
        # self.min_density = c.FLOAT_MAX

        # Defining subgraph's maximum density
        # self.max_density = -c.FLOAT_MAX

        # Initialize a zero-like array to hold the p.d.f. calculation
        pdf = np.zeros(self.n_nodes, dtype=np.float64)

        # For every possible node
        for i in range(self.n_nodes):

            n_pdf = 1.

            if len(self.nodes[i].shared_adjacency) > 0:

                # Initialize the number of p.d.f. calculations as 1
                n_pdf += len(self.nodes[i].shared_adjacency[:n_neighbours])

                distances = self.nodes[i].shared_adj_distances[:n_neighbours]

                # For every possible `k` value, calculate the p.d.f
                pdf[i] = np.sum(np.exp(-np.asfarray(distances) / (self.constant + c.EPSILON)))

            pdf[i] /= n_pdf

        # Applies subgraph's minimum density as p.d.f.'s value
        self.min_density = np.min(pdf)

        # Applies subgraph's maximum density as p.d.f.'s value
        self.max_density = np.max(pdf)

        # If subgraph's minimum density is the same as the maximum density
        if self.min_density == self.max_density:

            # For every possible node
            for i in range(self.n_nodes):
                # Applies node's density as maximum possible density
                self.nodes[i].density = c.MAX_DENSITY + 0.

                # Applies node's cost as maximum possible density - 1
                self.nodes[i].cost = c.MAX_DENSITY - 1.

        # If subgraph's minimum density is different than the maximum density
        else:
            # For every possible node
            for i in range(self.n_nodes):
                # Calculates the node's density
                self.nodes[i].density = (
                    (c.MAX_DENSITY - 1.) * (pdf[i] - self.min_density) /
                    (self.max_density - self.min_density)) + 1.

                # Calculates the node's cost
                self.nodes[i].cost = self.nodes[i].density - 1.

    def calculate_degree(self, n_neighbours):

        # Initialize a zero-like array to hold the p.d.f. calculation
        pdf = np.zeros(self.n_nodes, dtype=np.float64)

        # For every possible node
        for i in range(self.n_nodes):

            # if len(self.nodes[i].shared_adjacency) > 0:

            distances = np.asfarray(self.nodes[i].shared_adj_distances[:n_neighbours])

            weights = 1. - distances

            # For every possible `k` value, calculate the p.d.f
            pdf[i] = np.sum(weights)

        # Applies subgraph's minimum density as p.d.f.'s value
        self.min_density = np.min(pdf)

        # Applies subgraph's maximum density as p.d.f.'s value
        self.max_density = np.max(pdf)

        # If subgraph's minimum density is the same as the maximum density
        if self.min_density == self.max_density:
            # For every possible node
            for i in range(self.n_nodes):
                # Applies node's density as maximum possible density
                self.nodes[i].density = c.MAX_DENSITY + 0.

                # Applies node's cost as maximum possible density - 1
                self.nodes[i].cost = c.MAX_DENSITY - 1.

        # If subgraph's minimum density is different than the maximum density
        else:
            # For every possible node
            for i in range(self.n_nodes):
                # Calculates the node's density
                self.nodes[i].density = (
                    (c.MAX_DENSITY - 1) * (pdf[i] - self.min_density) / (
                        self.max_density - self.min_density)) + 1.

                # Calculates the node's cost
                self.nodes[i].cost = self.nodes[i].density - 1.

    def calculate_centrality(self, n_neighbors):

        dens = np.zeros(self.n_nodes, dtype=np.float64)

        # For every possible node
        for i in range(self.n_nodes):

            # Adjacent nodes corresponding to the current number of neighbors
            adjacents = self.nodes[i].shared_adjacency[:n_neighbors]

            # Distances corresponding to their adjacent nodes
            distances = self.nodes[i].shared_adj_distances[:n_neighbors]

            if not len(adjacents):
                continue

            # nn_list = adjacents.copy()
            # nn_list.insert(0, i)

            # id_map = {adj: j for j, adj in enumerate(nn_list)}

            # Calculate the adjacency relation restricted to the nearest neighbors
            # W = np.zeros((len(nn_list), len(nn_list)), dtype=np.float64)
            #
            # for idx in id_map:
            #
            #     adjs = self.nodes[idx].shared_adjacency[:n_neighbors]
            #     dists = self.nodes[idx].shared_adj_distances[:n_neighbors]
            #
            #     nn, nn_idx, _ = np.intersect1d(adjs, list(id_map.keys()), return_indices=True)
            #     indices = [id_map[idx] for idx in nn]
            #     # W[id_map[idx], indices] = 1. - np.array(dists)[nn_idx]
            #     W[id_map[idx], indices] = W.T[id_map[idx], indices] = 1. - np.array(dists)[nn_idx]

            W = np.zeros(shape=(len(adjacents) + 1, len(adjacents) + 1), dtype=np.float64)
            W[0, 1:] = W[1:, 0] = 1. - np.asfarray(distances, dtype=np.float64)

            # Normalized eigen decomposition of A: the eigenvector
            # corresponding to the maximum (principal) eigenvalue is chosen
            eig_vec, eig_val = g.norm_eigen_centrality(W)

            # Principal eigen value acts as a normalization factor
            dens[i] = np.abs(eig_vec[0]) / (eig_val + c.EPSILON)

        # Applies subgraph's minimum density as eigen centrality value
        self.min_density = np.min(dens)

        # Applies subgraph's maximum density as eigen centrality value
        self.max_density = np.max(dens)

        # Rescale nodes densities and costs to the interval [1, 1000]
        self.rescale_densities(dens)

    def create_arcs(self, k, distance_function, snn_function):

        # Creating an array of maximum distances
        max_distances = np.zeros(k)

        # Create the basic adjacency relation based on k-neighborhood
        for i in range(self.n_nodes):

            # Also make sure that it does not have any adjacent nodes (only for min cut)
            self.nodes[i].n_plateaus = 0

            i_neighbors, i_distances = [], []

            # Search for (k + 1) nearest neighbors
            for j in range(self.n_nodes):

                if i == j:
                    continue

                dist = distance_function(
                    self.nodes[i].features, self.nodes[j].features)

                i_distances.insert(0, dist)
                i_neighbors.insert(0, j)

            # Sort by distance and query only the top-k neighbors
            sorted_idx = np.argsort(i_distances)[:k]

            i_distances = np.array(i_distances)[sorted_idx].tolist()
            i_neighbors = np.array(i_neighbors)[sorted_idx].tolist()

            # Add current node as the 0-th neighbor
            # i_distances.insert(0, distance_function(self.nodes[i].features, self.nodes[i].features))
            # i_neighbors.insert(0, i)

            # Indices of neighbor nodes are added to node 'i' adjacency list
            # self.nodes[i].adjacency = i_neighbors[1:]
            self.nodes[i].adjacency = i_neighbors

            # Distances of neighbor nodes are added to no 'i' adjacency distance list
            # self.nodes[i].adj_distances = i_distances[1:]
            self.nodes[i].adj_distances = i_distances

        # Searching for shared nearest neighbors (SNN) to create a shared adjacency relation
        for i in range(self.n_nodes):

            for j in self.nodes[i].adjacency:

                i_neighbors = self.nodes[i].adjacency
                j_neighbors = self.nodes[j].adjacency

                # Verify if both nodes belong to each others k-neighborhood
                # if i in j_neighbors and j in i_neighbors:
                if i in j_neighbors:

                    # Measures SNN similarity between 'i' and 'j'
                    dist = snn_function(i_neighbors, j_neighbors)

                    # Node 'j' becomes a shared nearest neighbor of node 'i'
                    self.nodes[i].shared_adjacency.insert(0, j)

                    # Insert the similarity (derived distance) to the shared adjacency distances list
                    self.nodes[i].shared_adj_distances.insert(0, dist)

            tmp = self.nodes[i].shared_adj_distances.copy()

            # Sort shared distances and get their indices
            sorted_idx = np.argsort(tmp)

            self.nodes[i].shared_adj_distances = np.array(tmp)[sorted_idx].tolist()

            # Sort shared neighbors based on their corresponding sorted distances
            self.nodes[i].shared_adjacency = np.array(self.nodes[i].shared_adjacency)[sorted_idx].tolist()

            # Maximum distance among node 'i' shared neighbors
            max_dist = float(np.max(tmp)) if len(tmp) > 0 else 0.

            # Current graph density
            self.density = max(self.density, max_dist)

            # Define the node i radius
            self.nodes[i].radius = max_dist

            tmp = self.nodes[i].shared_adj_distances.copy()
            tmp.extend([0 for _ in range(k - len(tmp))])

            # Set the maximum distance with respect to each k shared neighbor
            max_distances = np.max([max_distances.tolist(), tmp], axis=0)

        if self.density < 1e-5:
            self.density = 1.

        return max_distances

    def create_ann_arcs(self, k, snn_function, ann_search):

        # Creating an array of maximum distances
        max_distances = np.zeros(k)

        # Create the basic adjacency relation based on k-neighborhood
        for i in range(self.n_nodes):

            # Also make sure that it does not have any adjacent nodes (only for min cut)
            self.nodes[i].n_plateaus = 0

            # Search for (k + 1) nearest neighbors
            i_neighbors, i_distances = ann_search.query(self.nodes[i].features, k + 1)

            # Indices of neighbor nodes are added to node 'i' adjacency list
            self.nodes[i].adjacency = list(i_neighbors[1:])

            # Distances of neighbor nodes are added to no 'i' adjacency distance list
            self.nodes[i].adj_distances = list(i_distances[1:])

        # Searching for shared nearest neighbors (SNN) to create a shared adjacency relation
        for i in range(self.n_nodes):

            for j in self.nodes[i].adjacency:

                i_neighbors = self.nodes[i].adjacency
                j_neighbors = self.nodes[j].adjacency

                # Verify if both nodes belong to each others k-neighborhood
                if i in j_neighbors:

                    # Measures SNN similarity between 'i' and 'j'
                    dist = snn_function(i_neighbors, j_neighbors)

                    # Node 'j' becomes a shared nearest neighbor of node 'i'
                    self.nodes[i].shared_adjacency.insert(0, j)

                    # Insert the similarity (derived distance) to the shared adjacency distances list
                    self.nodes[i].shared_adj_distances.insert(0, dist)

            tmp = self.nodes[i].shared_adj_distances.copy()

            # Sort shared distances and get their indices
            sorted_idx = np.argsort(tmp)

            self.nodes[i].shared_adj_distances = np.array(tmp)[sorted_idx].tolist()

            # Sort shared neighbors based on their corresponding sorted distances
            self.nodes[i].shared_adjacency = np.array(self.nodes[i].shared_adjacency)[sorted_idx].tolist()

            # Maximum distance among node 'i' shared neighbors
            max_dist = float(np.max(tmp)) if len(tmp) > 0 else 0.

            # Current graph density
            self.density = np.max([self.density, max_dist])

            # Define the node i radius
            self.nodes[i].radius = max_dist

            self.nodes[i].n_plateaus = 0

            tmp = self.nodes[i].shared_adj_distances.copy()
            tmp.extend([0 for _ in range(k - len(tmp))])

            # Set the maximum distance with respect to each k shared neighbor
            max_distances = np.max([max_distances.tolist(), tmp], axis=0)

        if self.density < 1e-5:
            self.density = 1.

        return max_distances

    def eliminate_maxima_height(self, height):
        """Eliminates maxima values in the subgraph that are below the inputted height.

        Args:
            height (float): Height's threshold.

        """

        logger.debug('Eliminating maxima above height = %s ...', height)

        # Checks if the height is non-negative
        if height > 0:
            # For every possible node
            for i in range(self.n_nodes):
                # Calculates its new cost
                self.nodes[i].cost = max(self.nodes[i].density - height, 0.)

        logger.debug('Maxima eliminated.')

    def eliminate_maxima_area(self, area):
        pass

    def eliminate_maxima_volume(self, volume):
        pass

    DENSITIES = {
        'pdf': calculate_pdf,
        'degree': calculate_degree,
        'eigen_centrality': calculate_centrality
    }
