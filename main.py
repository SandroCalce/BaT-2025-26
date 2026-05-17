import os
import tempfile
import shutil
import json
import networkx as nx
import numpy as np
import time as systime
import matplotlib.pyplot as plt

import datetime
from numpy.random import default_rng
from networkx import Graph

from src.qn_sim.core.topology import (
    gen_network_json,
    gen_traffic_mtx,
    gen_pair_queue,
    gen_request_time_list
)
from src.qn_sim.models.node import Node
from src.qn_sim.protocols.request import Request
from src.qn_sim.core.engine import Engine

from src.qn_sim.visualization.plots import Plots
from src.qn_sim.visualization.display_logger import DisplayLogger

# Network parameters
CONFIG = "network_customized.json"
GENERATE_NEW_NET = True
TRAFFIC_MATRIX = "traffic_matrix.json"
GENERATE_NEW_TRAFFIC = True
RANDOM_REQUESTS = True
NET_SIZE = 25
NET_TYPE = "grid"
CONTINUOUS_SCHEME = "adaptive"
SHORTCUT_STRATEGY = 'champion' # combined / champion
SHORTCUT_MIN_HOPS = 2

# Node parameters
MEMO_SIZE = 5  # default memory number per node
MEMO_LIFETIME = 100  # in units of simulation time step
ENTANGLEMENT_GEN_PROB = 0.1
ENTANGLEMENT_SWAP_PROB = 1
ADAPT_WEIGHT = 0.05

# Simulation parameters
SIM_SEED = 0
END_TIME = 1000000
NUM_TRIALS = 10
QUEUE_LEN = 10000
QUEUE_INT = 5
QUEUE_START = 1

if __name__ == "__main__":
    # Ensure necessary directories exist
    for folder in ["data", "stats", "logs"]:
        os.makedirs(folder, exist_ok=True)

    p = Plots()
    run_dir = p.output_dir

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
    plt.figure(figsize=(10, 8))
    nx.draw_networkx(G, pos)
    plt.title(f"Network Topology: {NET_TYPE} (Size: {NET_SIZE})")
    
    # Save the graph
    filepath = os.path.join(run_dir, f"network_topology.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Network topology graph saved to: {filepath}")

    G_temp = nx.Graph(graph_arr)
    stable_pos = nx.spring_layout(G_temp, seed=42)

    # Generate traffic matrix
    if GENERATE_NEW_TRAFFIC:
        traffic_mtx = gen_traffic_mtx(NET_SIZE, rng)
    else:
        tm = open(TRAFFIC_MATRIX)
        tm_json = json.load(tm)
        traffic_mtx = np.array(tm_json["matrix"])

    if RANDOM_REQUESTS:
        pair_queues_per_trial = [
            gen_pair_queue(traffic_mtx, NET_SIZE, QUEUE_LEN, rng, rng)
            for _ in range(NUM_TRIALS)
        ]
    else:
        fixed_pair_queue = [(16, 5), (13, 6), (15, 10), (11, 4), (10, 15), (16, 5), (13, 6), (15, 10), (11, 4), (10, 15)]
        pair_queues_per_trial = [fixed_pair_queue for _ in range(NUM_TRIALS)]

    pair_lookup = {
        trial_idx: dict(enumerate(queue))
        for trial_idx, queue in enumerate(pair_queues_per_trial)
    }

    multi_vis_available_graphs = []
    multi_vis_ondemand_graphs = []
    all_schemes = ["powerlaw", "uniform", "adaptive"]

    evaluation_data = {
        'request_completion': {},
        'shortcut_taken': {},
        'n_hops': {},
    }

    for scheme in all_schemes:
        CONTINUOUS_SCHEME = scheme
        evaluation_data['request_completion'][CONTINUOUS_SCHEME] = {}
        evaluation_data['shortcut_taken'][CONTINUOUS_SCHEME] = {}
        evaluation_data['n_hops'][CONTINUOUS_SCHEME] = {}
        print(scheme)
        for z in range(2):  # run simulation with and without changes to be able to compare them
            if z == 1:
                print("shortcut")
            evaluation_data['request_completion'][CONTINUOUS_SCHEME][z] = {}
            evaluation_data['n_hops'][CONTINUOUS_SCHEME][z] = {}
            latencies_list = []
            serve_times_list = []
            usage_pattern_list = []
            n_hops_list = []

            computed_shortcut_path = None
            shortcut_nodes = None
            if z == 1:
                G = Graph(graph_arr)
                shortcut_nodes = Engine.select_shortcut_nodes(G, strategy=SHORTCUT_STRATEGY, num_hops=SHORTCUT_MIN_HOPS)
                highest_node, second_highest_node = shortcut_nodes

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
                pair_queue = pair_queues_per_trial[trial]
                # Generate request submission time list with constant interval
                time_list = gen_request_time_list(QUEUE_START, QUEUE_LEN, interval=QUEUE_INT)
                # Generate request stack
                request_stack = [Request(time, pair, uid=i) for i, (time, pair) in
                                 enumerate(zip(time_list, pair_queue))]

                logger = None
                #ENABLE LOGGER HERE:
                # if trial == 0:  # Only debug the first trial
                #     logger = DisplayLogger(graph_arr, nodes_layout_pos=stable_pos, shortcut_path=computed_shortcut_path, scheme=scheme, z=z)

                """ Run Simulation"""
                engine = Engine(graph_arr, nodes, request_stack, z, seed=SIM_SEED, logger=logger, shortcut_nodes=shortcut_nodes)
                latencies, serve_times, congestion, request_complete_times, entanglement_usage_pattern, n_hops, process_data = \
                    engine.run(END_TIME)

                if logger:
                    logger.root.destroy()

                latencies_list.append(latencies)
                serve_times_list.append(serve_times)
                usage_pattern_list.append(entanglement_usage_pattern)
                n_hops_list.append(n_hops)
                print("Finished trial {} of {}".format(trial + 1, NUM_TRIALS))
                print(f"Number of request completed {len(n_hops)}")
                evaluation_data['request_completion'][CONTINUOUS_SCHEME][z][trial] = process_data['request_completion']
                evaluation_data['n_hops'][CONTINUOUS_SCHEME][z][trial] = {
                    uid: hops for (uid, _, _), hops in zip(process_data['request_completion'], n_hops)
                }
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
            filename = os.path.join(p.data_dir, "data_" + CONTINUOUS_SCHEME + "_" + str(z) + ".json")
            data = {
                "latencies": latencies_list,
                "n_hops": n_hops_list,
                "service_times": serve_times_list,
                "average_latencies": latencies_avg.tolist(),
                "average_service_times": serve_times_avg.tolist(),
                "accumulated_available_patterns": available_accum,
                "accumulated_ondemand_patterns": ondemand_accum
            }
            if os.path.exists(filename):
                os.remove(filename)

            # Atomic write to prevent partial file
            with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
                json.dump(data, tmp)
                tmp.flush()
                tmp_path = tmp.name

            shutil.move(tmp_path, filename)

    p.plot_timing_schemes(evaluation_data['request_completion'])
    p.plot_win_percentage(evaluation_data['request_completion'])
    p.plot_service_win_percentage(evaluation_data['request_completion'])
    p.plot_serve_time_band(evaluation_data['request_completion'])
    p.plot_completion_time_band(evaluation_data['request_completion'])
    p.plot_percentage_improvement(evaluation_data['request_completion'])
    p.plot_hop_distribution(evaluation_data['n_hops'])
    p.plot_serve_time_cdf(evaluation_data['request_completion'])
    p.plot_speedup_vs_hops(evaluation_data['request_completion'], evaluation_data['n_hops'])
    p.plot_od_speedup_heatmap(evaluation_data['request_completion'], pair_lookup, net_size=NET_SIZE)
    p.plot_speedup_by_cluster(evaluation_data['request_completion'], pair_lookup, graph_arr)
    number_of_complete_requests = {}
    for key in evaluation_data['request_completion'].keys():
        number_of_complete_requests[key] = {}
        for trial, item in evaluation_data['request_completion'][key][1].items():
            number_of_complete_requests[key][trial] = len(item)

    p.plot_shortcut_usage(evaluation_data['shortcut_taken'], number_of_complete_requests)
