"""Contains graph elements."""

import numpy as np


class Node(object):
    """Holds a node current sum, vertexes and path"""
    def __init__(self, vertexes, node_sum):
        self.vertexes = vertexes
        self.sum = node_sum

    def update_sum(self, node_sum):
        """Update sum with new value.

        Args:
            node_sum(int): The sum to update the node.
        """
        self.sum = node_sum

    def get_bow(self, i):
        """Get the i'th bow."""
        return self.vertexes[i]

    def get_sum(self):
        """Get the node's current sum"""
        return self.sum


class Layer(object):
    """Holds on layer in the graph."""
    def __init__(self, layer_weight_matrix):
        """Constructor.

        Args:
            layer_weight_matrix(np.ndarray): A square matrix, where the first row matches
                the vertexes of the first node.
        """
        node_num = layer_weight_matrix.shape[0]
        self.nodes = {}
        self.node_names = ['node_{}'.format(i) for i in xrange(node_num)]
        for i in xrange(node_num):
            self.nodes[self.node_names[i]] = Node((layer_weight_matrix[i]), 0)

    def update_nodes_sum(self, layer_sum):
        """Update all nodes in the layer.

        Args:
            layer_sum(np.array): states long array.
        """
        for i, node_name in enumerate(self.node_names):
            self.nodes[node_name].update_sum(layer_sum[i])

    def get_node(self, i):
        """Get the i'th layer in the graph.

        Args:
            i(int): The layer to retreive.

        Returns:
            Layer. A layer object.

        """
        return self.nodes['node_{}'.format(i)]


class Graph(object):
    """Holds all layer in the graph."""
    def __init__(self, graph_weights):
        """Constructor.

        Args:
            graph_weights(np.ndarray): Contains all the vertexes weights in the graph.
                graph_weights[i, :, :] - contains all the weights for layer i.
                graph_weights[i, j, :] - contains all the weights for layer i, that comes
                out of state j.
                graph_weights[i, :, l] - contains all the weights for layer i, that comes
                in to state l.
        """
        self.num_layers = graph_weights.shape[0]
        self.num_states = graph_weights.shape[1]
        # Adding a dummy layer.
        graph_weights = np.concatenate([graph_weights,
                                        np.zeros([1, self.num_states, self.num_states])])

        self.num_layers += 1
        self.layers = {}
        self.layers_names = ['layer_{}'.format(i) for i in xrange(self.num_layers)]
        for i in xrange(self.num_layers):
            self.layers[self.layers_names[i]] = Layer(graph_weights[i])

    def get_layer(self, i):
        """Get the i'th layer in the graph.

        Args:
            i(int): The layer to retreive.

        Returns:
            Layer. A layer object.

        """
        return self.layers['layer_{}'.format(i)]

    def get_num_layers(self):
        """Get the number of layers in the graph."""
        return self.num_layers

    def get_num_states(self):
        """Get the states in a layers in the graph."""
        return self.num_states
