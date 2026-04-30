import time as systime
import os
import datetime

from Plots import Plots

import numpy as np
import requests
from matplotlib import pyplot as plt
import networkx as nx
import tempfile
import shutil

from simulation_core import *
from hardware import *
from protocols import *
from DisplayLogger import DisplayLogger

# Network parameters
CONFIG = "network_customized.json"
GENERATE_NEW_NET = True
TRAFFIC_MATRIX = "traffic_matrix.json"
GENERATE_NEW_TRAFFIC = True
RANDOM_REQUESTS = True
NET_SIZE = 20
NET_TYPE = "as_net"
CONTINUOUS_SCHEME = "adaptive"

# Node parameters
MEMO_SIZE = 5  # default memory number per node
MEMO_LIFETIME = 100  # in units of simulation time step
ENTANGLEMENT_GEN_PROB = 0.1
ENTANGLEMENT_SWAP_PROB = 1
ADAPT_WEIGHT = 0.05

# Simulation parameters
SIM_SEED = 0
END_TIME = 10000
NUM_TRIALS = 100
QUEUE_LEN = 100
QUEUE_INT = 5
QUEUE_START = 1


def run_simulation(graph_arr, nodes, request_stack, end_time, z, seed=0, logger=None):
    shuffle_rng = default_rng(seed)

    sim_time = 0

    # metrics
    latencies = []  # keep track of latencies for each request to get completed
    serve_times = []  # keep track of times to serve each request
    congestion = []  # keep track of number of incomplete requests at the end of each time step
    request_complete_times = []  # keep track of when each request is completed
    entanglement_usage_pattern = {"available": [],
                                  "ondemand": []}  # keep track of entanglement usage pattern for every request

    requests_to_serve = []  # keep track of incomplete requests, in case new request comes in before previous request is completed
    entanglement_available = []  # keep track of entanglement links from route nodes when a request is submitted
    entanglement_ondemand = []  # keep track of entanglement links generated on demand to complete a request
    n_hops = []
    # track current request and related info
    next_request_to_submit = request_stack.pop(0)
    current_request = None
    origin_node = None
    destination_node = None
    route = []

    process_data = {
        'request_completion': [],
        'shortcut_taken': 0
    }

    shortcut_path_nodes = []
    shortcut_active = False

    sc_start = None
    sc_end = None
    routing_graph = graph_arr.copy()

    def activate_request(req):
        """Calculates route and updates node protocols for a request starting NOW."""
        # 1. Calculate path based on current (live) graph state
        new_route = req.get_path(routing_graph, nodes)
        req.route = new_route

        # 2. Update node protocols (Left/Right neighbors)
        local_entanglement_available = []

        for i, label in enumerate(new_route):
            node = nodes[label]

            left_neighbors_to_connect = new_route[:i]
            right_neighbors_to_connect = new_route[i + 1:]
            nodes[label].left_neighbors_to_connect.append(left_neighbors_to_connect)
            nodes[label].right_neighbors_to_connect.append(right_neighbors_to_connect)

            links_available = []
            links_used = []

            # get current links
            for other_label, count in node.entanglement_link_nums.items():
                if count > 0:
                    links_available.append(other_label)
                    if other_label not in left_neighbors_to_connect:
                        links = [(label, other_label)] * count
                        local_entanglement_available.extend(links)

            # get links used for request
            if i > 0:
                links_used.append(new_route[i - 1])
            if i < (len(new_route) - 1):
                links_used.append(new_route[i + 1])
            node.generation_protocol.update_dist(links_available, links_used)

        # 3. Record entanglement usage
        entanglement_usage_pattern["available"].append(local_entanglement_available)

        return req, new_route, nodes[new_route[0]], nodes[new_route[-1]]

    if z == 1:
        # prepare shortcut
        G = Graph(graph_arr)
        between = nx.betweenness_centrality(G, normalized=True, endpoints=True)
        degree = nx.degree_centrality(G)
        combined_centrality = {}
        for node in between:
            combined_centrality[node] = between[node] + degree[node]

        # Sort nodes by their combined centrality in descending order
        sorted_combined_centrality = sorted(combined_centrality.items(), key=lambda x: x[1], reverse=True)

        # Get the highest and second-highest values
        highest_node, highest_value = sorted_combined_centrality[0]
        second_highest_node, second_highest_value = sorted_combined_centrality[1]
        req = Request(sim_time, (highest_node, second_highest_node), -1)
        print((highest_node, highest_value), (second_highest_node, second_highest_value))
        # find path to pre entangle
        shortcut_path = req.get_path(graph_arr, nodes)

        shortcut_path_nodes = [nodes[x] for x in shortcut_path]
        sc_start = shortcut_path_nodes[0]
        sc_end = shortcut_path_nodes[-1]

        # Building the shortcut left and right neighbours for smart connection
        for i, label in enumerate(shortcut_path):
            sc_node = nodes[label]

            sc_left_neighbours_to_connect = shortcut_path[:i]
            sc_right_neighbours_to_connect = shortcut_path[i + 1:]
            sc_node.sc_left_neighbors_to_connect.append(sc_left_neighbours_to_connect)
            sc_node.sc_right_neighbors_to_connect.append(sc_right_neighbours_to_connect)

        print([x.label for x in shortcut_path_nodes])
        # ... existing code in run_simulation ...

    while sim_time < end_time:
        current_events = []
        # check if memories expired
        for node in nodes:
            for memory in node.memories:
                expire_time = memory.entangled_memory["expire_time"]
                if expire_time is not None and expire_time <= sim_time:

                    node.memo_expire(memory)
                    current_events.append(f"Node {node.label} memory expired")

        if z == 1:
            shortcut_active = False
            for mem in sc_start.memories:
                if mem.entangled_memory["node"] == sc_end:
                    shortcut_active = True
                    break
            sc_start_idx = sc_start.label
            sc_end_idx = sc_end.label
            if shortcut_active:
                routing_graph[sc_start_idx][sc_end_idx] = 1
                routing_graph[sc_end_idx][sc_start_idx] = 1
                current_events.append(f"[SHORTCUT] Shortcut ACTIVE: {sc_start_idx} <-> {sc_end_idx}")
            else:
                routing_graph[sc_start_idx][sc_end_idx] = graph_arr[sc_start_idx][sc_end_idx]
                routing_graph[sc_end_idx][sc_start_idx] = graph_arr[sc_end_idx][sc_start_idx]
                current_events.append(f"[SHORTCUT] Shortcut INACTIVE: Building {sc_start_idx} <-> {sc_end_idx}")

            if not shortcut_active:
                for i in range(len(shortcut_path_nodes)):
                    sc_node = shortcut_path_nodes[i]

                    direct_sc_left = None
                    direct_sc_left_node = None
                    direct_sc_right = None
                    direct_sc_right_node = None

                    sc_left_neighbors = sc_node.sc_left_neighbors_to_connect[0]
                    if len(sc_left_neighbors) > 0:
                        direct_sc_left = sc_left_neighbors[-1]
                        direct_sc_left_node = nodes[direct_sc_left]
                    sc_right_neighbors = sc_node.sc_right_neighbors_to_connect[0]
                    if len(sc_right_neighbors) > 0:
                        direct_sc_right = sc_right_neighbors[0]
                        direct_sc_right_node = nodes[direct_sc_right]
                    if sc_node is sc_start:
                        sc_right_entanglement_link_nums = [sc_node.entanglement_link_nums[i] for i in sc_right_neighbors]

                        if not any(sc_right_entanglement_link_nums):
                            success = sc_node.create_link_with_priority(sim_time, direct_sc_right_node)
                            if success:
                                current_events.append(
                                    f"[SHORTCUT] Link created: Node {sc_node.label} <-> Node {direct_sc_right}")
                            else:
                                current_events.append(
                                    f"[SHORTCUT] Link FAILED: Node {sc_node.label} <-> Node {direct_sc_right}")

                    elif sc_node is sc_end:
                        sc_left_entanglement_link_nums = [sc_node.entanglement_link_nums[i] for i in sc_left_neighbors]

                        if not any(sc_left_entanglement_link_nums):
                            success = sc_node.create_link_with_priority(sim_time, direct_sc_left_node)
                            if success:
                                current_events.append(f"[SHORTCUT] Link created: Node {direct_sc_left} <-> Node {sc_node.label}")
                            else:
                                current_events.append(f"[SHORTCUT] Link FAILED: Node {direct_sc_left} <-> Node {sc_node.label}")

                    else:
                        sc_left_entanglement_link_nums = [sc_node.entanglement_link_nums[i] for i in sc_left_neighbors]
                        sc_right_entanglement_link_nums = [sc_node.entanglement_link_nums[i] for i in sc_right_neighbors]
                        if not any(sc_left_entanglement_link_nums):
                            success = sc_node.create_link_with_priority(sim_time, direct_sc_left_node)
                            if success:
                                current_events.append(
                                    f"[SHORTCUT] Link created: Node {direct_sc_left} <-> Node {sc_node.label}")
                            else:
                                current_events.append(
                                    f"[SHORTCUT] Link FAILED: Node {direct_sc_left} <-> Node {sc_node.label}")
                        elif not any(sc_right_entanglement_link_nums):
                            success = sc_node.create_link_with_priority(sim_time, direct_sc_right_node)
                            if success:
                                current_events.append(
                                    f"[SHORTCUT] Link created: Node {sc_node.label} <-> Node {direct_sc_right}")
                            else:
                                current_events.append(
                                    f"[SHORTCUT] Link FAILED: Node {sc_node.label} <-> Node {direct_sc_right}")
                        else:
                            sc_right_reversed = list(reversed(sc_right_neighbors))
                            sc_right_nums_reversed = list(reversed(sc_right_entanglement_link_nums))
                            sc_left_most = next(
                                (label for num, label in zip(sc_left_entanglement_link_nums, sc_left_neighbors) if
                                 num > 0), sc_node.label)
                            sc_right_most = next(
                                (label for num, label in zip(sc_right_nums_reversed, sc_right_reversed) if
                                 num > 0), sc_node.label)

                            assert sc_left_most != sc_node.label
                            assert sc_right_most != sc_node.label

                            sc_left_most_node = nodes[sc_left_most]
                            sc_right_most_node = nodes[sc_right_most]

                            sc_left_memory = next(
                                (mem for mem in sc_node.memories if mem.entangled_memory["node"] == sc_left_most_node),
                                None)
                            sc_right_memory = next(
                                (mem for mem in sc_node.memories if mem.entangled_memory["node"] == sc_right_most_node),
                                None)

                            swap_success = sc_node.swap(sc_left_memory, sc_right_memory, sc_start, sc_end, sim_time)
                            if swap_success:
                                current_events.append(
                                    f"[SHORTCUT] Swap SUCCESS at Node {sc_node.label} ({sc_left_most}<->{sc_right_most})")
                            else:
                                current_events.append(
                                    f"[SHORTCUT] Swap FAILED at Node {sc_node.label} ({sc_left_most}<->{sc_right_most})")

        # determine if a new request is submitted to the network
        if sim_time == next_request_to_submit.submit_time:
            current_events.append(f"New Request: {next_request_to_submit.pair}")
            # submit request
            requests_to_serve.append(next_request_to_submit)

            # get new request
            if len(request_stack) > 0:
                next_request_to_submit = request_stack.pop(0)

        if current_request is None and len(requests_to_serve) > 0:
            current_request = requests_to_serve[0]
            current_request.start_time = sim_time

            current_request, route, origin_node, destination_node = activate_request(current_request)

        nodes_shuffled = list(nodes)
        shuffle_rng.shuffle(nodes_shuffled)

        # call function to run node (entanglement generation) protocol
        for node in nodes_shuffled:
            n = node.label

            if n not in route:
                node.create_random_link(sim_time)

            else:
                # get neighbor information in the path
                direct_right = None
                direct_right_node = None
                direct_left = None
                direct_left_node = None
                left_neighbors = node.left_neighbors_to_connect[0]
                if len(left_neighbors) > 0:
                    direct_left = left_neighbors[-1]
                    direct_left_node = nodes[direct_left]
                right_neighbors = node.right_neighbors_to_connect[0]
                if len(right_neighbors) > 0:
                    direct_right = right_neighbors[0]
                    direct_right_node = nodes[direct_right]

                # determine if the node is the origin node of the route
                if node is origin_node:
                    right_entanglement_link_nums = [node.entanglement_link_nums[i] for i in right_neighbors]
                    # if no entanglement link with right neighbors, create link with direct right neighbor on demand
                    if not any(right_entanglement_link_nums):
                        success = node.create_link_with_priority(sim_time, direct_right_node)
                        if success:
                            current_events.append(f"[ROUTE] Link on-demand: Node {node.label} <-> Node {direct_right}")
                        else:
                            current_events.append(f"[ROUTE] Link FAILED: Node {node.label} <-> Node {direct_right}")
                        entanglement_ondemand.append((node.label, direct_right))

                # determine if the node is the destination node of the route
                elif node is destination_node:
                    left_entanglement_link_nums = [node.entanglement_link_nums[i] for i in left_neighbors]
                    # if no entanglement link with left neighbors, create link with direct left neighbor on demand
                    if not any(left_entanglement_link_nums):
                        success = node.create_link_with_priority(sim_time, direct_left_node)
                        if success:
                            current_events.append(f"[ROUTE] Link on-demand: Node {direct_left} <-> Node {node.label}")
                        else:
                            current_events.append(f"[ROUTE] Link FAILED: Node {direct_left} <-> Node {node.label}")
                        entanglement_ondemand.append((direct_left, node.label))

                # otherwise the node is in the middle of the route
                else:

                    left_entanglement_link_nums = [node.entanglement_link_nums[i] for i in left_neighbors]
                    right_entanglement_link_nums = [node.entanglement_link_nums[i] for i in right_neighbors]

                    # if no entanglement link with left neighbors, create link with direct left neighbor on demand
                    if not any(left_entanglement_link_nums):
                        success = node.create_link_with_priority(sim_time, direct_left_node)
                        if success:
                            current_events.append(f"[ROUTE] Link on-demand: Node {direct_left} <-> Node {node.label}")
                        else:
                            current_events.append(f"[ROUTE] Link FAILED: Node {direct_left} <-> Node {node.label}")
                        entanglement_ondemand.append((direct_left, node.label))

                    # if no entanglement link with right neighbors, create link with direct right neighbor on demand
                    elif not any(right_entanglement_link_nums):
                        success = node.create_link_with_priority(sim_time, direct_right_node)
                        if success:
                            current_events.append(f"[ROUTE] Link on-demand: Node {node.label} <-> Node {direct_right}")
                        else:
                            current_events.append(f"[ROUTE] Link FAILED: Node {node.label} <-> Node {direct_right}")
                        entanglement_ondemand.append((node.label, direct_right))

                    # if both sides have entanglement links, try swapping
                    else:
                        # choose memories with rightmost and leftmost entanglement
                        # find leftmost and rightmost entangled nodes
                        right_reversed = list(reversed(right_neighbors))
                        right_nums_reversed = list(reversed(right_entanglement_link_nums))
                        leftmost = next((label for num, label in zip(left_entanglement_link_nums, left_neighbors)
                                         if num > 0), n)
                        rightmost = next((label for num, label in zip(right_nums_reversed, right_reversed)
                                          if num > 0), n)
                        assert leftmost != n
                        assert rightmost != n

                        leftmost_node = nodes[leftmost]
                        rightmost_node = nodes[rightmost]

                        left_memory = next((mem for mem in node.memories
                                            if mem.entangled_memory["node"] == leftmost_node), None)
                        right_memory = next((mem for mem in node.memories
                                             if mem.entangled_memory["node"] == rightmost_node), None)

                        swap_success = node.swap(left_memory, right_memory, sc_start, sc_end, sim_time)
                        if swap_success:
                            current_events.append(f"[ROUTE] Swap SUCCESS: Node {node.label} ({leftmost}<->{rightmost})")
                        else:
                            current_events.append(f"[ROUTE] Swap FAILED: Node {node.label} ({leftmost}<->{rightmost})")

        # determine if the desired entanglement is established
        if current_request is not None:
            for memory in origin_node.memories:
                # check if we have memory entangled with destination
                if memory.entangled_memory["node"] == destination_node:
                    current_events.append(f"Completed Request: {current_request.pair}")
                    # record latency and completion time
                    latency = int(sim_time - current_request.submit_time)
                    serve_time = int(sim_time - current_request.start_time)
                    latencies.append(latency)
                    serve_times.append(serve_time)
                    request_complete_times.append(sim_time)
                    n_hops.append(len(current_request.route) - 1)
                    # record entanglement links generated on demand and reset entanglement_ondemand
                    entanglement_usage_pattern["ondemand"].append(entanglement_ondemand)
                    entanglement_ondemand = []

                    process_data['request_completion'].append((current_request.uid, sim_time))
                    if z == 1:
                        if sc_start.label in route and sc_end.label in route:
                            process_data['shortcut_taken'] += 1

                    # clean left and right neighbors_to_connect information for nodes in current route
                    for node_label in route:
                        nodes[node_label].left_neighbors_to_connect.pop(0)
                        nodes[node_label].right_neighbors_to_connect.pop(0)
                    # expire memories
                    origin_node.memo_expire(memory)

                    requests_to_serve.pop(0)

                    current_request = None
                    route = []
                    origin_node = None
                    destination_node = None

                    break

        congestion.append(len(requests_to_serve))

        # check if no more requests
        if len(request_stack) == 0 and len(requests_to_serve) == 0:
            break

        if logger:
            logger.log_state(
                sim_time,
                nodes,
                requests_to_serve,
                current_events,
                shortcut_active=shortcut_active,
                current_route=route,
                active_request=current_request  # <--- NEW PARAMETER
            )

        sim_time += 1

    # average latencies (over time) and return
    return [latencies, serve_times, congestion, request_complete_times, entanglement_usage_pattern, n_hops,
            process_data]


if __name__ == "__main__":
    # Setup rng
    rng = default_rng(SIM_SEED)

    # Generate network
    default_memos = [MEMO_SIZE] * NET_SIZE
    if GENERATE_NEW_NET:
        graph_arr = gen_network_json(CONFIG, NET_SIZE, NET_TYPE, SIM_SEED)
        memo_sizes = default_memos
    else:
        fh = open(CONFIG)
        topo = json.load(fh)
        graph_arr = np.array(topo["array"])
        memo_sizes = np.array(topo.get("memo_sizes", default_memos))
        assert graph_arr.shape == (NET_SIZE, NET_SIZE)
        assert len(memo_sizes) == NET_SIZE
    G = nx.Graph(graph_arr)
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos)
    plt.show()

    G_temp = nx.Graph(graph_arr)
    stable_pos = nx.spring_layout(G_temp, seed=42)

    # Generate traffic matrix
    if GENERATE_NEW_TRAFFIC:
        traffic_mtx = gen_traffic_mtx(NET_SIZE, rng)
    else:
        tm = open(TRAFFIC_MATRIX)
        tm_json = json.load(tm)
        traffic_mtx = np.array(tm_json["matrix"])

    multi_vis_available_graphs = []
    multi_vis_ondemand_graphs = []
    all_schemes = ["powerlaw", "uniform", "adaptive"]

    evaluation_data = {
        'request_completion': {},
        'shortcut_taken': {},
    }

    for scheme in all_schemes:
        CONTINUOUS_SCHEME = scheme
        evaluation_data['request_completion'][CONTINUOUS_SCHEME] = {}
        evaluation_data['shortcut_taken'][CONTINUOUS_SCHEME] = {}
        print(scheme)
        for z in range(2):  # run simulation with and without changes to be able to compare them
            if z == 1:
                print("shortcut")
            evaluation_data['request_completion'][CONTINUOUS_SCHEME][z] = {}
            latencies_list = []
            serve_times_list = []
            usage_pattern_list = []
            n_hops_list = []

            computed_shortcut_path = None
            if z == 1:
                G = Graph(graph_arr)
                between = nx.betweenness_centrality(G, normalized=True, endpoints=True)
                degree = nx.degree_centrality(G)
                combined_centrality = {}
                for node in between:
                    combined_centrality[node] = between[node] + degree[node]

                sorted_combined_centrality = sorted(combined_centrality.items(), key=lambda x: x[1], reverse=True)
                highest_node, highest_value = sorted_combined_centrality[0]
                second_highest_node, second_highest_value = sorted_combined_centrality[1]

                # Create temporary nodes to get the path
                temp_nodes = [
                    Node(i, memo_size, MEMO_LIFETIME, ENTANGLEMENT_GEN_PROB, ENTANGLEMENT_SWAP_PROB, graph_arr,
                         seed=i) for i, memo_size in enumerate(memo_sizes)]
                req = Request(0, (highest_node, second_highest_node), -1)
                computed_shortcut_path = req.get_path(graph_arr, temp_nodes)
                print(f"Computed shortcut path: {computed_shortcut_path}")

            tick = systime.time()
            for trial in range(NUM_TRIALS):
                # set nodes
                seed_start = NET_SIZE * trial
                nodes = [Node(i, memo_size, MEMO_LIFETIME, ENTANGLEMENT_GEN_PROB, ENTANGLEMENT_SWAP_PROB, graph_arr,
                              seed=seed_start + i)
                         for i, memo_size in enumerate(memo_sizes)]
                for node in nodes:
                    other_nodes = nodes[:]
                    other_nodes.remove(node)
                    node.set_other_nodes(other_nodes)
                    node.set_generation_protocol(CONTINUOUS_SCHEME, ADAPT_WEIGHT)

                # Generate request node pair queue
                if RANDOM_REQUESTS:
                    pair_queue = gen_pair_queue(traffic_mtx, NET_SIZE, QUEUE_LEN, rng, rng)
                else:
                    pair_queue = [(16, 5), (13, 6), (15, 10), (11, 4), (10, 15), (16, 5), (13, 6), (15, 10), (11, 4), (10, 15)]
                # Generate request submission time list with constant interval
                time_list = gen_request_time_list(QUEUE_START, QUEUE_LEN, interval=QUEUE_INT)
                # Generate request stack
                request_stack = [Request(time, pair, uid=i) for i, (time, pair) in
                                 enumerate(zip(time_list, pair_queue))]

                logger = None
                #ENABLE LOGGER HERE:
                if trial == 0:  # Only debug the first trial
                    logger = DisplayLogger(graph_arr, nodes_layout_pos=stable_pos, shortcut_path=computed_shortcut_path, scheme=scheme, z=z)

                """ Run Simulation"""
                latencies, serve_times, congestion, request_complete_times, entanglement_usage_pattern, n_hops, process_data = \
                    run_simulation(graph_arr, nodes, request_stack, END_TIME, z, seed=SIM_SEED, logger=logger)

                if logger:
                    logger.root.destroy()

                latencies_list.append(latencies)
                serve_times_list.append(serve_times)
                usage_pattern_list.append(entanglement_usage_pattern)
                n_hops_list.append(n_hops)
                print("Finished trial {} of {}".format(trial + 1, NUM_TRIALS))
                print(f"Number of request completed {len(n_hops)}")
                evaluation_data['request_completion'][CONTINUOUS_SCHEME][z][trial] = process_data['request_completion']
                if z == 1:
                    evaluation_data['shortcut_taken'][CONTINUOUS_SCHEME][trial] = process_data['shortcut_taken']
            sim_time = systime.time() - tick
            print("Total simulation time: ", sim_time)
            print("Average time per trial: ", sim_time / NUM_TRIALS)

            num_latencies = min([len(latencies_list[i]) for i in range(NUM_TRIALS)])
            num_serve_times = min([len(serve_times_list[i]) for i in range(NUM_TRIALS)])
            num_requests = min(num_latencies,
                               num_serve_times)  # num_latencies and num_serve_times should be equal in principle
            latencies_avg = np.zeros(num_requests)
            serve_times_avg = np.zeros(num_requests)

            for i in range(NUM_TRIALS):
                latencies_avg += np.array(latencies_list[i][:num_requests])

            for i in range(NUM_TRIALS):
                serve_times_avg += np.array(serve_times_list[i][:num_requests])

            latencies_avg = latencies_avg / NUM_TRIALS
            serve_times_avg = serve_times_avg / NUM_TRIALS

            print(f"Finished Sim-Run {z}")

            # construct error
            low_percentile = np.zeros(num_latencies)
            high_percentile = np.zeros(num_latencies)
            low_percentile_serve = np.zeros(num_latencies)
            high_percentile_serve = np.zeros(num_latencies)
            for i in range(num_latencies):
                low_percentile[i] = np.percentile([ll[i] for ll in latencies_list], 5)
                high_percentile[i] = np.percentile([ll[i] for ll in latencies_list], 95)
                low_percentile_serve[i] = np.percentile([ll[i] for ll in serve_times_list], 5)
                high_percentile_serve[i] = np.percentile([ll[i] for ll in serve_times_list], 95)

            # entanglement usage pattern information
            available_patterns = [usage_pattern_list[i]["available"] for i in range(NUM_TRIALS)]
            ondemand_patterns = [usage_pattern_list[i]["ondemand"] for i in range(NUM_TRIALS)]
            available_accum = [[] for i in range(num_requests)]
            ondemand_accum = [[] for i in range(num_requests)]
            for i in range(num_requests):
                for pattern in available_patterns:
                    available_accum[i] += pattern[i]
                for pattern in ondemand_patterns:
                    ondemand_accum[i] += pattern[i]

            # choose the first, the last and the middle requests' patterns for visualization
            vis_available_patterns = [available_accum[0], available_accum[round(num_requests / 2)], available_accum[-1]]
            vis_ondemand_patterns = [ondemand_accum[0], ondemand_accum[round(num_requests / 2)], ondemand_accum[-1]]
            vis_available_graphs = []
            vis_ondemand_graphs = []
            for pattern in vis_available_patterns:
                G_vis = nx.Graph(graph_arr)
                nx.set_edge_attributes(G_vis, 0, "available")
                # nx.set_edge_attributes(G_vis, 0, "ondemand")
                for pair in pattern:
                    if (pair[0], pair[1]) not in G_vis.edges():
                        G_vis.add_edge(pair[0], pair[1], available=1)
                    else:
                        G_vis[pair[0]][pair[1]]["available"] += 1
                vis_available_graphs.append(G_vis)
            multi_vis_available_graphs.append(vis_available_graphs)

            for pattern in vis_ondemand_patterns:
                G_vis = nx.Graph(graph_arr)
                # nx.set_edge_attributes(G_vis, 0, "available")
                nx.set_edge_attributes(G_vis, 0, "ondemand")
                for pair in pattern:
                    if (pair[0], pair[1]) not in G_vis.edges():
                        G_vis.add_edge(pair[0], pair[1], ondemand=1)
                    else:
                        G_vis[pair[0]][pair[1]]["ondemand"] += 1
                vis_ondemand_graphs.append(G_vis)
            multi_vis_ondemand_graphs.append(vis_ondemand_graphs)

            # save data
            filename = "data_" + CONTINUOUS_SCHEME + "_" + str(z) + ".json"
            data = {
                "latencies": latencies_list,
                "n_hops": n_hops_list,
                "service_times": serve_times_list,
                "average_latencies": latencies_avg.tolist(),
                "average_service_times": serve_times_avg.tolist(),
                "accumulated_available_patterns": available_accum,
                "accumulated_ondemand_patterns": ondemand_accum
            }
            os.remove(filename)

            # Atomic write to prevent partial file
            with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
                json.dump(data, tmp)
                tmp.flush()
                tmp_path = tmp.name

            shutil.move(tmp_path, filename)

    p = Plots()

    p.plot_timing_schemes(evaluation_data['request_completion'])
    p.plot_win_percentage(evaluation_data['request_completion'])
    number_of_complete_requests = {}
    for key in evaluation_data['request_completion'].keys():
        number_of_complete_requests[key] = {}
        for trial, item in evaluation_data['request_completion'][key][1].items():
            number_of_complete_requests[key][trial] = len(item)

    p.plot_shortcut_usage(evaluation_data['shortcut_taken'], number_of_complete_requests)

    """
    # statistics visualization
    requests_latencies = np.arange(num_latencies)
    requests_serve_times = np.arange(num_serve_times)

    fig = plt.figure(figsize=(7, 7))

    ax1 = plt.subplot(211)
    ax1.plot(requests_latencies, latencies_avg)
    ax1.set_title("average request latencies")
    ax1.fill_between(requests_latencies, high_percentile, low_percentile, alpha=0.4)

    ax2 = plt.subplot(212)
    ax2.plot(requests_serve_times, serve_times_avg)
    ax2.set_title("average times to serve requests")
    ax2.fill_between(requests_serve_times, high_percentile_serve, low_percentile_serve, alpha=0.4)

    plt.xlabel("request number")
    plt.tight_layout()
    plt.show()

    # patterns visualization on graphs
    for vis_available_graphs in multi_vis_available_graphs:
        for Graph in vis_available_graphs:
            edges = Graph.edges()
            avails = [Graph[u][v]["available"] for u,v in edges]
            nx.draw_networkx_nodes(Graph, pos)
            nx.draw_networkx_labels(Graph, pos)
            edges_drawn = nx.draw_networkx_edges(Graph, pos, edge_color=avails, width=2, edge_cmap=plt.cm.Greens, edge_vmin=0)
            plt.colorbar(edges_drawn)
            plt.axis('off')
            plt.show()
    for vis_ondemand_graphs in multi_vis_ondemand_graphs:
        for Graph in vis_ondemand_graphs:
            edges = Graph.edges()
            ondemands = [Graph[u][v]["ondemand"] for u,v in edges]
            nx.draw_networkx_nodes(Graph, pos)
            nx.draw_networkx_labels(Graph, pos)
            edges_drawn = nx.draw_networkx_edges(Graph, pos, edge_color=ondemands, width=2, edge_cmap=plt.cm.Reds, edge_vmin=0)
            plt.colorbar(edges_drawn)
            plt.axis('off')
            plt.show()

    import json
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from collections import defaultdict

    # Load both JSON files
    datasets = {}
    for i in range(2):
        for scheme in all_schemes:
            with open(f"data_{scheme}_{i}.json", "r") as f:
                datasets[f"{scheme} {i + 1}"] = json.load(f)

    # Convert data to a flat list of dictionaries for pandas
    records = []
    for run_name, data in datasets.items():
        latencies_batches = data["latencies"]
        n_hops_batches = data["n_hops"]

        for lat_batch, hop_batch in zip(latencies_batches, n_hops_batches):
            for latency, hops in zip(lat_batch, hop_batch):
                records.append({
                    "Latency [ms]": latency,
                    "Number of Hops": hops,
                    "Run": run_name
                })

    # Create a DataFrame
    df = pd.DataFrame(records)

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Number of Hops", y="Latency [ms]", hue="Run", data=df, width=0.8, dodge=True, palette="Set2")
    plt.ylim(0, 400)
    plt.title("Latency Distribution by Number of Hops (Comparison of Runs)")
    plt.grid(True)
    handles, labels = plt.gca().get_legend_handles_labels()
    custom_labels = []
    for x in all_schemes:
        custom_labels.append(x)
    for x in all_schemes:
        custom_labels.append(x + " shortcut")
    plt.legend(handles, custom_labels, title="Run", loc="upper left")
    plt.tight_layout()
    plt.show()

    # Load data
    with open("data_adaptive_1.json", "r") as f:
        data = json.load(f)

    # Build records: each (trial index, hop value)
    records = []
    for trial_idx, hop_list in enumerate(data["n_hops"]):
        for hop in hop_list:
            records.append({"Trial": trial_idx, "Hops": hop})

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Count how many requests per (Trial, Hops)
    grouped = df.groupby(["Trial", "Hops"]).size().reset_index(name="Count")

    # Plot grouped barplot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=grouped, x="Hops", y="Count", hue="Trial", dodge=True)
    plt.title("Number of Requests per Hop per Trial")
    plt.xlabel("Number of Hops")
    plt.ylabel("Number of Requests")
    plt.legend(title="Trial", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
    """