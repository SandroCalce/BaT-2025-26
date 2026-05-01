from networkx import Graph, shortest_path
from .base import GenerationProtocol

class UniformGenerationProtocol(GenerationProtocol):
    """Class representing protocol to generate entanglement links.

    This protocol has probabilities following a uniform distribution.
    """

    def __init__(self, node, network, distance=1):
        """Constructor of entanglement generation protocol instance.

        Args:
            node (Node): host node.
            network (np.ndarray): adjacency array for the network.
            distance (int): max distance to nodes to select.
        """

        super().__init__(node)
        G = Graph(network)
        possible = [n.label for n in node.other_nodes if len(shortest_path(G, node.label, n.label)) - 1 <= distance]
        prob = 1 / len(possible)
        self.prob_dist = {n: prob for n in possible}
        self.starting_prob_dist = self.prob_dist


class PowerLawGenerationProtocol(GenerationProtocol):
    """Class representing protocol to generate entanglement links.

    This protocol has probabilities following an power law (power -1) distribution, with closer nodes more likely.
    """

    def __init__(self, node, network):
        """Constructor of entanglement generation protocol instance.

        Args:
            node (Node): host node.
            network (np.ndarray): adjacency array for the network.
        """

        super().__init__(node)
        G = Graph(network)
        self.prob_dist = {n.label: 1 / len(shortest_path(G, node.label, n.label)) for n in node.other_nodes}
        total = sum(self.prob_dist.values())
        for label in self.prob_dist:
            self.prob_dist[label] /= total
        self.starting_prob_dist = self.prob_dist


class AdaptiveGenerationProtocol(GenerationProtocol):
    """Class representing protocol to generate entanglement links.

    This protocol will update the probabilities adaptively based on network traffic.
    """

    def __init__(self, node, adapt_param, neighbors):
        """Constructor of entanglement generation protocol instance.

        Args:
            node (Node): node hosting the protocol instance.
            adapt_param (float): sets alpha parameter for adaptive update of probabilities.
            neighbors (List[int]): list of labels for neighboring nodes.
        """

        super().__init__(node)
        self.alpha = adapt_param
        self.neighbors = neighbors

        init_prob = 1 / len(neighbors)
        self.prob_dist = {neighbor: init_prob for neighbor in neighbors}
        self.starting_prob_dist = self.prob_dist

    def update_dist(self, links_available, links_used):
        """Method to update the probability distribution adaptively.

        Called when a request is sent to the network.

        Args:
            links_available (List[int]): entanglement links available before the request is submitted.
            links_used (List[int]): entanglement links used to complete the request.
        """

        avail = set(links_available) & set(self.neighbors)
        used = set(links_used) & set(self.neighbors)

        S = avail & used
        T = used - avail
        not_used = set(self.neighbors) - used

        # increase probability for links in T
        if len(T) > 0:
            sum_st = sum([self.prob_dist[i] for i in (S | T)])
            new_prob_increase = (self.alpha / len(T)) * (1 - sum_st)
            for t in T:
                self.prob_dist[t] += new_prob_increase

        # decrease probability for links not in T or S
        if len(not_used) > 0:
            sum_st_new = sum([self.prob_dist[i] for i in used])
            new_prob = (1 - sum_st_new) / len(not_used)
            for i in not_used:
                self.prob_dist[i] = new_prob
