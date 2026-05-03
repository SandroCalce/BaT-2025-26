import numpy as np
from numpy.random import default_rng
import networkx as nx
from networkx import Graph

# Core dependencies
from ..protocols.request import Request

class SimulationEngine:
    """
    Simulation Engine for Quantum Network Entanglement.
    
    Orchestrates the discrete-time simulation of entanglement generation, 
    swapping, and shortcut mechanisms.
    """

    def __init__(self, graph_arr, nodes, request_stack, z, seed=0, logger=None):
        """
        Initialize the simulation engine.

        Args:
            graph_arr (np.ndarray): The adjacency matrix of the network.
            nodes (List[Node]): List of pre-configured Node objects.
            request_stack (List[Request]): List of requests to be processed.
            z (int): Shortcut flag (1 to enable shortcut logic).
            seed (int): Random seed for simulation reproducibility.
            logger (DisplayLogger, optional): GUI logger for visualization.
        """
        self.graph_arr = graph_arr
        self.nodes = nodes
        self.request_stack = request_stack
        self.z = z
        self.seed = seed
        self.logger = logger
        self.shuffle_rng = default_rng(seed)

        # Simulation State
        self.sim_time = 0
        self.requests_to_serve = []
        self.current_request = None
        self.route = []
        self.origin_node = None
        self.destination_node = None
        
        # Shortcut State
        self.shortcut_path_nodes = []
        self.shortcut_active = False
        self.sc_start = None
        self.sc_end = None
        self.routing_graph = graph_arr.copy()

        # Metrics
        self.latencies = []
        self.serve_times = []
        self.congestion = []
        self.request_complete_times = []
        self.entanglement_usage_pattern = {"available": [], "ondemand": []}
        self.entanglement_ondemand = []
        self.n_hops = []
        self.process_data = {
            'request_completion': [],
            'shortcut_taken': 0
        }

        if self.z == 1:
            self._initialize_shortcut()

    def _initialize_shortcut(self):
        """Pre-calculates and sets up the shortcut infrastructure."""
        G = Graph(self.graph_arr)
        between = nx.betweenness_centrality(G, normalized=True, endpoints=True)
        degree = nx.degree_centrality(G)
        combined_centrality = {node: between[node] + degree[node] for node in between}

        # Find two most central nodes
        sorted_nodes = sorted(combined_centrality.items(), key=lambda x: x[1], reverse=True)
        highest_node = sorted_nodes[0][0]
        second_highest_node = sorted_nodes[1][0]
        
        # Create a dummy request to find the path for the shortcut
        temp_req = Request(0, (highest_node, second_highest_node), -1)
        shortcut_path = temp_req.get_path(self.graph_arr, self.nodes)
        
        self.shortcut_path_nodes = [self.nodes[x] for x in shortcut_path]
        self.sc_start = self.shortcut_path_nodes[0]
        self.sc_end = self.shortcut_path_nodes[-1]

        # Configure nodes for the shortcut
        for i, label in enumerate(shortcut_path):
            sc_node = self.nodes[label]
            sc_node.sc_left_neighbors_to_connect.append(shortcut_path[:i])
            sc_node.sc_right_neighbors_to_connect.append(shortcut_path[i + 1:])

    def _activate_request(self, req):
        """Calculates route and updates node protocols for a request starting NOW."""
        new_route = req.get_path(self.routing_graph, self.nodes)
        req.route = new_route

        local_entanglement_available = []

        for i, label in enumerate(new_route):
            node = self.nodes[label]
            
            left_neighbors = new_route[:i]
            right_neighbors = new_route[i + 1:]
            node.left_neighbors_to_connect.append(left_neighbors)
            node.right_neighbors_to_connect.append(right_neighbors)

            # Record metrics
            links_available = []
            for other_label, count in node.entanglement_link_nums.items():
                if count > 0:
                    links_available.append(other_label)
                    if other_label not in left_neighbors:
                        local_entanglement_available.extend([(label, other_label)] * count)

            # Update protocol distribution
            links_used = []
            if i > 0: links_used.append(new_route[i - 1])
            if i < (len(new_route) - 1): links_used.append(new_route[i + 1])
            node.generation_protocol.update_dist(links_available, links_used)

        self.entanglement_usage_pattern["available"].append(local_entanglement_available)
        
        return new_route, self.nodes[new_route[0]], self.nodes[new_route[-1]]

    def run(self, end_time):
        """Executes the simulation loop until end_time or all requests are served."""
        next_request_to_submit = self.request_stack.pop(0) if self.request_stack else None

        while self.sim_time < end_time:
            current_events = []
            
            # 1. Handle Memory Expiration
            self._handle_memory_expirations(current_events)

            # 2. Shortcut Logic (if enabled)
            if self.z == 1:
                self._manage_shortcut(current_events)

            # 3. Handle New Request Submission
            if next_request_to_submit and self.sim_time == next_request_to_submit.submit_time:
                current_events.append(f"New Request: {next_request_to_submit.pair}")
                self.requests_to_serve.append(next_request_to_submit)
                next_request_to_submit = self.request_stack.pop(0) if self.request_stack else None

            # 4. Activate Next Request in Queue
            if self.current_request is None and self.requests_to_serve:
                self.current_request = self.requests_to_serve[0]
                self.current_request.start_time = self.sim_time
                self.route, self.origin_node, self.destination_node = self._activate_request(self.current_request)

            # 5. Execute Node Protocols (Generation & Swapping)
            self._execute_node_steps(current_events)

            # 6. Check for Request Completion
            if self.current_request and self._check_completion(current_events):
                # Request finished, loop continues to next step or breaks if done
                pass

            self.congestion.append(len(self.requests_to_serve))

            # 7. Check Global Exit Condition
            if not next_request_to_submit and not self.requests_to_serve:
                break

            # 8. Visualization
            if self.logger:
                self.logger.log_state(
                    self.sim_time, self.nodes, self.requests_to_serve, current_events,
                    shortcut_active=self.shortcut_active, current_route=self.route,
                    active_request=self.current_request
                )

            self.sim_time += 1

        return self._prepare_results()

    def _handle_memory_expirations(self, events):
        for node in self.nodes:
            for memory in node.memories:
                expire_time = memory.entangled_memory["expire_time"]
                if expire_time is not None and expire_time <= self.sim_time:
                    node.memo_expire(memory)
                    events.append(f"Node {node.label} memory expired")

    def _manage_shortcut(self, events):
        self.shortcut_active = False
        for mem in self.sc_start.memories:
            if mem.entangled_memory["node"] == self.sc_end:
                self.shortcut_active = True
                break
        
        sc_start_idx, sc_end_idx = self.sc_start.label, self.sc_end.label
        if self.shortcut_active:
            self.routing_graph[sc_start_idx][sc_end_idx] = 1
            self.routing_graph[sc_end_idx][sc_start_idx] = 1
            events.append(f"[SHORTCUT] Shortcut ACTIVE: {sc_start_idx} <-> {sc_end_idx}")
        else:
            self.routing_graph[sc_start_idx][sc_end_idx] = self.graph_arr[sc_start_idx][sc_end_idx]
            self.routing_graph[sc_end_idx][sc_start_idx] = self.graph_arr[sc_end_idx][sc_start_idx]
            events.append(f"[SHORTCUT] Shortcut INACTIVE: Building {sc_start_idx} <-> {sc_end_idx}")
            self._build_shortcut(events)

    def _build_shortcut(self, events):
        for i, sc_node in enumerate(self.shortcut_path_nodes):
            sc_left_neighbors = sc_node.sc_left_neighbors_to_connect[0]
            sc_right_neighbors = sc_node.sc_right_neighbors_to_connect[0]
            
            direct_sc_left = self.nodes[sc_left_neighbors[-1]] if sc_left_neighbors else None
            direct_sc_right = self.nodes[sc_right_neighbors[0]] if sc_right_neighbors else None

            if sc_node is self.sc_start:
                if not any(sc_node.entanglement_link_nums[n] for n in sc_right_neighbors):
                    success = sc_node.create_link_with_priority(self.sim_time, direct_sc_right)
                    events.append(f"[SHORTCUT] Link {'created' if success else 'FAILED'}: {sc_node.label} <-> {direct_sc_right.label}")
            elif sc_node is self.sc_end:
                if not any(sc_node.entanglement_link_nums[n] for n in sc_left_neighbors):
                    success = sc_node.create_link_with_priority(self.sim_time, direct_sc_left)
                    events.append(f"[SHORTCUT] Link {'created' if success else 'FAILED'}: {direct_sc_left.label} <-> {sc_node.label}")
            else:
                self._intermediate_shortcut_step(sc_node, sc_left_neighbors, sc_right_neighbors, direct_sc_left, direct_sc_right, events)

    def _intermediate_shortcut_step(self, node, left_n, right_n, d_left, d_right, events):
        if not any(node.entanglement_link_nums[n] for n in left_n):
            success = node.create_link_with_priority(self.sim_time, d_left)
            events.append(f"[SHORTCUT] Link {'created' if success else 'FAILED'}: {d_left.label} <-> {node.label}")
        elif not any(node.entanglement_link_nums[n] for n in right_n):
            success = node.create_link_with_priority(self.sim_time, d_right)
            events.append(f"[SHORTCUT] Link {'created' if success else 'FAILED'}: {node.label} <-> {d_right.label}")
        else:
            # Try Swapping
            leftmost = next((label for label in left_n if node.entanglement_link_nums[label] > 0), node.label)
            rightmost = next((label for label in reversed(right_n) if node.entanglement_link_nums[label] > 0), node.label)
            
            mem_l = next((m for m in node.memories if m.entangled_memory["node"] == self.nodes[leftmost]), None)
            mem_r = next((m for m in node.memories if m.entangled_memory["node"] == self.nodes[rightmost]), None)
            swap_sucess, output_str = node.swap(mem_l, mem_r, self.sc_start, self.sc_end, self.sim_time)
            if swap_sucess:
                events.append(f"[SHORTCUT] Swap SUCCESS at {node.label} ({leftmost}<->{rightmost}) {output_str}")
            else:
                events.append(f"[SHORTCUT] Swap FAILED at {node.label} ({leftmost}<->{rightmost}) {output_str}")

    def _execute_node_steps(self, events):
        shuffled_nodes = list(self.nodes)
        self.shuffle_rng.shuffle(shuffled_nodes)

        for node in shuffled_nodes:
            if node.label not in self.route:
                node.create_random_link(self.sim_time)
            else:
                self._process_route_node(node, events)

    def _process_route_node(self, node, events):
        left_n = node.left_neighbors_to_connect[0]
        right_n = node.right_neighbors_to_connect[0]
        d_left = self.nodes[left_n[-1]] if left_n else None
        d_right = self.nodes[right_n[0]] if right_n else None

        if node is self.origin_node:
            if not any(node.entanglement_link_nums[n] for n in right_n):
                if node.create_link_with_priority(self.sim_time, d_right):
                    events.append(f"[ROUTE] Link on-demand Left: {node.label} <-> {d_right.label}")
                    self.entanglement_ondemand.append((node.label, d_right.label))
        elif node is self.destination_node:
            if not any(node.entanglement_link_nums[n] for n in left_n):
                if node.create_link_with_priority(self.sim_time, d_left):
                    events.append(f"[ROUTE] Link on-demand Right: {d_left.label} <-> {node.label}")
                    self.entanglement_ondemand.append((d_left.label, node.label))
        else:
            # Middle node
            if not any(node.entanglement_link_nums[n] for n in left_n):
                if node.create_link_with_priority(self.sim_time, d_left):
                    events.append(f"[ROUTE] Link on-demand Left: {d_left.label} <-> {node.label}")
                    self.entanglement_ondemand.append((d_left.label, node.label))
            elif not any(node.entanglement_link_nums[n] for n in right_n):
                if node.create_link_with_priority(self.sim_time, d_right):
                    events.append(f"[ROUTE] Link on-demand Right: {node.label} <-> {d_right.label}")
                    self.entanglement_ondemand.append((node.label, d_right.label))
            else:
                # Swap
                leftmost = next((label for label in left_n if node.entanglement_link_nums[label] > 0), node.label)
                rightmost = next((label for label in reversed(right_n) if node.entanglement_link_nums[label] > 0), node.label)
                mem_l = next((m for m in node.memories if m.entangled_memory["node"] == self.nodes[leftmost]), None)
                mem_r = next((m for m in node.memories if m.entangled_memory["node"] == self.nodes[rightmost]), None)
                swap_sucess, output_str = node.swap(mem_l, mem_r, self.sc_start, self.sc_end, self.sim_time)
                if swap_sucess:
                    events.append(f"[ROUTE] Swap SUCCESS: {node.label} ({leftmost}<->{rightmost}) {output_str}")
                else:
                    events.append(f"[ROUTE] Swap FAILED at {node.label} ({leftmost}<->{rightmost}) {output_str}")

    def _check_completion(self, events):
        for memory in self.origin_node.memories:
            if memory.entangled_memory["node"] == self.destination_node:
                events.append(f"Completed Request: {self.current_request.pair}")
                
                # Record Metrics
                latency = int(self.sim_time - self.current_request.submit_time)
                serve_time = int(self.sim_time - self.current_request.start_time)
                self.latencies.append(latency)
                self.serve_times.append(serve_time)
                self.request_complete_times.append(self.sim_time)
                self.n_hops.append(len(self.current_request.route) - 1)
                self.entanglement_usage_pattern["ondemand"].append(self.entanglement_ondemand)
                self.entanglement_ondemand = []
                self.process_data['request_completion'].append((self.current_request.uid, self.sim_time, serve_time))
                
                if self.z == 1 and self.sc_start.label in self.route and self.sc_end.label in self.route:
                    self.process_data['shortcut_taken'] += 1

                # Cleanup
                for node_label in self.route:
                    self.nodes[node_label].left_neighbors_to_connect.pop(0)
                    self.nodes[node_label].right_neighbors_to_connect.pop(0)
                
                self.origin_node.memo_expire(memory)
                self.requests_to_serve.pop(0)
                self.current_request = None
                self.route = []
                self.origin_node = None
                self.destination_node = None
                return True
        return False

    def _prepare_results(self):
        return [
            self.latencies,
            self.serve_times,
            self.congestion,
            self.request_complete_times,
            self.entanglement_usage_pattern,
            self.n_hops,
            self.process_data
        ]
