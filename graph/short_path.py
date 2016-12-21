# Shortest path implementation.

import numpy as np
from itertools import izip

from graph_elements import Graph

BIG_NUMBER = 2**32
ALL_VALID_STATE = -1


class FindShortestPath(object):
    """Holds functionality for finding the shortest path on the graph."""

    VERY_LARGE_NUMBER = 2**30

    def __init__(self, graph, debug=False, verbose=False):
        self.graph = graph
        self.num_layres = self.graph.get_num_layers()
        self.num_states = self.graph.get_num_states()
        self.bows_from_node = self.num_states
        self.verbose = verbose

    def run_forward_path(self):
        """Run graph forward path."""
        best_paths = []
        for i in xrange(self.num_layres - 1):
            left_layer = self.graph.get_layer(i)

            new_sums, paths = self.get_one_forward_path_step_new_sum(left_layer)
            if self.verbose:
                print 'Min costs in layer {} is: {}'.format(i + 1, new_sums)
            best_paths.append(paths)
            self.update_graph(new_sums, i+1)

        if self.verbose:
            print '-------------------------------------------'
            print 'Finished at {}'.format(np.argmin(new_sums))
            print "Path min cost is: {}".format(min(new_sums))
        best_path = self.backward_path(best_paths, np.argmin(new_sums))
        return best_path

    def backward_path(self, best_paths, finish_index):
        """Get best path."""
        current_finish_index = finish_index
        path = [finish_index]
        for best_bows in best_paths[::-1]:
            for bow in best_bows:
                if bow[1] == current_finish_index:
                    prev = bow[0]

                    current_finish_index = prev
                    path.append(prev)
                    break

        return path[::-1]

    def update_graph(self, new_sum, layer_index):
        """Update grpah sums layer with new sums.

        Args:
            new_sum(list): A list with the new sums to update
                the layer.

            layer_index(int): The layer to update.
        """
        self.graph.get_layer(layer_index).update_nodes_sum(new_sum)

    def get_one_forward_path_step_new_sum(self, left_layer):
        """Run one forward step.

        Args:
            left_layer(Layer): The graph left layer.

        Returns:
            tuple. consists of two lists: an array with the new sums to be updated,
                and the stats that led to those sums.
        """
        new_sums = []
        best_left_node_indexs = []
        for right_node_index in xrange(self.num_states):
            smallest_sum = self.VERY_LARGE_NUMBER
            best_left_node_index = -1

            for left_node_index in xrange(self.bows_from_node):
                left_node = left_layer.get_node(left_node_index)
                left_node_current_sum = left_node.get_sum()
                left_bow = left_node.get_bow(right_node_index)

                new_sum = left_bow + left_node_current_sum
                if new_sum < smallest_sum:
                    smallest_sum = new_sum
                    best_left_node_index = left_node_index

            best_left_node_indexs.append([best_left_node_index, right_node_index])
            new_sums.append(smallest_sum)

        return new_sums, best_left_node_indexs


def get_shortest_path(graph_weights, verbose=False):
    """Get the shortest path on the graph.

    Args:
        graph_weights(np.ndarray): Contains all the bows weights in the graph.
          graph_weights[i, :, :] - contains all the weights for layer i.

    Returns:
        list. contains the shortest states to pass.
    """
    graph = Graph(graph_weights)
    shortest = FindShortestPath(graph, verbose=verbose)
    return shortest.run_forward_path()


def get_regularized_weights(weights, forbidden_states):
    """Get weighs with very high weights on the forbidden states.

    Note:
        Each state[i] > forbidden_states[i] will be assigned high also

    Args:
        weights(np.ndarray): Contains all the bows weights in the graph.
            graph_weights[i, :, :] - contains all the weights for layer i.
            graph_weights[i, ,j, :] - contains all the weights for layer i, that enters state j.
        forbidden_states(list): a list of states not to pass.
            A ALL_VALID_STATE value will indicate all of the states are
            valid in this layer.

    Returns:
        np.ndarray. The new weights matrix.
    """

    # Validating the zero state is always allowed.
    # Validating states values are in the permitted values.
    assert (len(forbidden_states) == weights.shape[0])
    assert not (0 in forbidden_states)
    assert (max(forbidden_states) < weights.shape[1])

    regularized_weights = np.copy(weights)
    for layer_ind, forbidden_state in enumerate(forbidden_states):
        if forbidden_state != ALL_VALID_STATE:
            regularized_weights[layer_ind, :, forbidden_state:] = BIG_NUMBER

    return regularized_weights


def is_forbidden_states_valid(best_states, forbidden_states):
    """Validate that there were no invalid passes in the graph.

    Args:
        best_states(list): The list of the best states.
        forbidden_states(list): The list of the states not to pass int.

    Returns:
        bool. True is the state is valid.
    """
    assert (len(best_states) - 1 == len(forbidden_states))
    # Starts from one because we don't want the first state, only the ones we enter.
    for best, forbidden in izip(best_states[1:], forbidden_states):
        if forbidden != ALL_VALID_STATE:
            if not best < forbidden:
                return False

    return True


def test():
    print 'test '
    layres = 3
    states = 3
    WHEIGTS = np.zeros([layres, states, states])
    WHEIGTS[0, 0, 0] = 0
    WHEIGTS[0, 0, 1] = 1
    WHEIGTS[0, 0, 2] = 3
    WHEIGTS[0, 1, 0] = 4
    WHEIGTS[0, 1, 1] = 5
    WHEIGTS[0, 1, 2] = 6
    WHEIGTS[0, 2, 0] = 1
    WHEIGTS[0, 2, 1] = 2
    WHEIGTS[0, 2, 2] = 3
    WHEIGTS[1, 0, 0] = 7
    WHEIGTS[1, 0, 1] = 8
    WHEIGTS[1, 0, 2] = 1
    WHEIGTS[1, 1, 0] = 9
    WHEIGTS[1, 1, 1] = 5
    WHEIGTS[1, 1, 2] = 3
    WHEIGTS[1, 2, 0] = 10
    WHEIGTS[1, 2, 1] = 1
    WHEIGTS[1, 2, 2] = 2
    WHEIGTS[2, 0, 0] = 1
    WHEIGTS[2, 0, 1] = 3
    WHEIGTS[2, 0, 2] = 7
    WHEIGTS[2, 1, 0] = 8
    WHEIGTS[2, 1, 1] = 1
    WHEIGTS[2, 1, 2] = 9
    WHEIGTS[2, 2, 0] = 10
    WHEIGTS[2, 2, 1] = 2
    WHEIGTS[2, 2, 2] = 13

    best_paths = get_shortest_path(WHEIGTS)
    assert best_paths[:4] == [0, 0, 2, 1]
    print 'best paths: {}.'.format(best_paths)
    print 'test passed.'

if __name__ == '__main__':
    test()
