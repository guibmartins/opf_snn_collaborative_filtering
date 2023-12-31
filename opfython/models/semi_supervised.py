"""Semi-Supervised Optimum-Path Forest.
"""

import time

import numpy as np

import opfython.utils.constants as c
import opfython.utils.logging as l
from opfython.core import Heap, Node, Subgraph
from opfython.models import SupervisedOPF

logger = l.get_logger(__name__)


class SemiSupervisedOPF(SupervisedOPF):
    """A SemiSupervisedOPF which implements the semi-supervised version of OPF classifier.

    References:
        W. P. Amorim, A. X. Falcão and M. H. Carvalho. Semi-supervised Pattern Classification Using Optimum-Path Forest.
        27th SIBGRAPI Conference on Graphics, Patterns and Images (2014).

    """

    def __init__(self, distance='log_squared_euclidean', pre_computed_distance=None):
        """Initialization method.

        Args:
            distance (str): An indicator of the distance metric to be used.
            pre_computed_distance (str): A pre-computed distance file for feeding into OPF.

        """

        logger.info('Overriding class: SupervisedOPF -> SemiSupervisedOPF.')

        super(SemiSupervisedOPF, self).__init__(distance, pre_computed_distance)

        logger.info('Class overrided.')

    def fit(self, X_train, Y_train, X_unlabeled, I_train=None):
        """Fits data in the semi-supervised classifier.

        Args:
            X_train (np.array): Array of training features.
            Y_train (np.array): Array of training labels.
            X_unlabeled (np.array): Array of unlabeled features.
            I_train (np.array): Array of training indexes.

        """

        logger.info('Fitting semi-supervised classifier ...')

        start = time.time()

        # Creating a subgraph
        self.subgraph = Subgraph(X_train, Y_train, I_train)

        # Finding prototypes
        self._find_prototypes()

        # Gather current number of nodes
        current_n_nodes = self.subgraph.n_nodes

        for i, feature in enumerate(X_unlabeled):
            node = Node(current_n_nodes + i, 0, feature)

            self.subgraph.nodes.append(node)

        # Creating a minimum heap
        h = Heap(size=self.subgraph.n_nodes)

        for i in range(self.subgraph.n_nodes):
            if self.subgraph.nodes[i].status == c.PROTOTYPE:
                # If yes, it does not have predecessor nodes
                self.subgraph.nodes[i].pred = c.NIL

                # Its predicted label is the same as its true label
                self.subgraph.nodes[i].predicted_label = self.subgraph.nodes[i].label

                # Its cost equals to zero
                h.cost[i] = 0

                # Inserts the node into the heap
                h.insert(i)

            else:
                # Its cost equals to maximum possible value
                h.cost[i] = c.FLOAT_MAX

        while not h.is_empty():
            # Removes a node
            p = h.remove()

            # Appends its index to the ordered list
            self.subgraph.idx_nodes.append(p)

            # Gathers its cost
            self.subgraph.nodes[p].cost = h.cost[p]

            for q in range(self.subgraph.n_nodes):
                if p != q:
                    if h.cost[p] < h.cost[q]:
                        if self.pre_computed_distance:
                            weight = self.pre_distances[self.subgraph.nodes[p]
                                                        .idx][self.subgraph.nodes[q].idx]

                        else:
                            weight = self.distance_fn(self.subgraph.nodes[p].features, self.subgraph.nodes[q].features)

                        # The current cost will be the maximum cost between the node's and its weight (arc)
                        current_cost = np.maximum(h.cost[p], weight)

                        if current_cost < h.cost[q]:
                            # `q` node has `p` as its predecessor
                            self.subgraph.nodes[q].pred = p

                            # And its predicted label is the same as `p`
                            self.subgraph.nodes[q].predicted_label = self.subgraph.nodes[p].predicted_label

                            # As we may have unlabeled nodes, make sure that `q` label equals to `q` predicted label
                            self.subgraph.nodes[q].label = self.subgraph.nodes[q].predicted_label

                            # Updates the heap `q` node and the current cost
                            h.update(q, current_cost)

        # The subgraph has been properly trained
        self.subgraph.trained = True

        end = time.time()

        train_time = end - start

        logger.info('Semi-supervised classifier has been fitted.')
        logger.info('Training time: %s seconds.', train_time)
