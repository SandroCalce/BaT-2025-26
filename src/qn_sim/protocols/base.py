from abc import ABC

class GenerationProtocol(ABC):
    """Class representing protocol to generate entanglement links.

    Attributes:
        node (Node): node hosting the protocol instance.
        prob_dist (Dict[int, float]): probability distribution to select direct neighbors to generate entanglement.
    """

    def __init__(self, node):
        """Constructor of entanglement generation protocol instance.

        Args:
            node (Node): node hosting the protocol instance.
        """
        self.node = node
        self.prob_dist = {}
        self.starting_prob_dist = {}

    def reset(self):
        self.prob_dist = self.starting_prob_dist

    def update_dist(self, links_available, links_used):
        pass

    def choose_link(self):
        """Method to choose a link to attempt entanglement.

        Returns:
            int: label of node chosen for entanglement
        """

        choices = list(self.prob_dist.keys())
        probs = list(self.prob_dist.values())
        return self.node.rng.choice(choices, p=probs)
