from networkx import Graph, shortest_path

class Request:
    """Class representing single requests for generating entanglement between two nodes.

    Attributes:
        uid (int): Unique identifier for the request
        submit_time (int): time to submit the request
        start_time (int): time when the network starts to serve the request
        pair (Tuple[int, int]): keeps track of labels of origin and destination nodes of the request
        route (List[int]): route of nodes for entanglement connection to complete the request
    """

    def __init__(self, submit_time, pair, uid):
        """Constructor of a request instance.

        Args:
            submit_time (int): time to submit the request
            pair (Tuple[int, int]): keeps track of labels of origin and destination nodes of the request
        """
        self.uid = uid
        self.submit_time = submit_time
        self.start_time = submit_time  # start time is no earlier than submit time
        self.pair = pair
        self.route = None

    def get_path(self, network, nodes):
        """Get optimal path to service request.

        Uses local best effort algorithm based on number of existing entanglement links.

        Args:
            network (numpy.ndarray): Adjacency matrix for the network.
            nodes (List[Node]): List of node objects for the network, contains current entanglement info.

        Returns:
            List[int]: Optimal path as list of node labels.
        """

        G = Graph(network)
        end = self.pair[1]
        u_curr = self.pair[0]
        path = [u_curr]

        while u_curr != end:
            node = nodes[u_curr]
            virtual_neighbors = [n for n, count in node.entanglement_link_nums.items() if count > 1]
            if len(virtual_neighbors) == 0:
                u = shortest_path(G, u_curr, end)[1]
            else:
                distances = [len(shortest_path(G, v, end)) - 1 for v in virtual_neighbors]
                minimum_distance = min(distances)

                u = virtual_neighbors[distances.index(minimum_distance)]
                if len(shortest_path(G, u_curr, end)) <= len(shortest_path(G, u, end)):
                    u = shortest_path(G, u_curr, end)[1]
            """
            zweiter graph testen mit wormhole link für shortest path
            """

            path.append(u)
            u_curr = u

        return path
