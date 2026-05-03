import os
import json
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

class LegacyPlots:
    """
    Contains legacy plotting functions that were previously in main.py.
    """
    
    @staticmethod
    def plot_statistics(num_latencies, latencies_avg, high_percentile, low_percentile, 
                        num_serve_times, serve_times_avg, high_percentile_serve, low_percentile_serve):
        """
        Statistics visualization for latencies and service times.
        """
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

    @staticmethod
    def plot_patterns_on_graphs(multi_vis_available_graphs, multi_vis_ondemand_graphs, pos):
        """
        Patterns visualization on graphs for available and on-demand entanglements.
        """
        # patterns visualization on graphs
        for vis_available_graphs in multi_vis_available_graphs:
            for Graph in vis_available_graphs:
                edges = Graph.edges()
                avails = [Graph[u][v]["available"] for u, v in edges]
                nx.draw_networkx_nodes(Graph, pos)
                nx.draw_networkx_labels(Graph, pos)
                edges_drawn = nx.draw_networkx_edges(Graph, pos, edge_color=avails, width=2, edge_cmap=plt.cm.Greens, edge_vmin=0)
                plt.colorbar(edges_drawn)
                plt.axis('off')
                plt.show()
        
        for vis_ondemand_graphs in multi_vis_ondemand_graphs:
            for Graph in vis_ondemand_graphs:
                edges = Graph.edges()
                ondemands = [Graph[u][v]["ondemand"] for u, v in edges]
                nx.draw_networkx_nodes(Graph, pos)
                nx.draw_networkx_labels(Graph, pos)
                edges_drawn = nx.draw_networkx_edges(Graph, pos, edge_color=ondemands, width=2, edge_cmap=plt.cm.Reds, edge_vmin=0)
                plt.colorbar(edges_drawn)
                plt.axis('off')
                plt.show()

    @staticmethod
    def plot_latency_distribution_comparison(all_schemes, data_dir="data"):
        """
        Latency Distribution by Number of Hops (Comparison of Runs).
        """
        # Load both JSON files
        datasets = {}
        for i in range(2):
            for scheme in all_schemes:
                file_path = os.path.join(data_dir, f"data_{scheme}_{i}.json")
                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
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
        if df.empty:
            print("No data found for latency distribution comparison.")
            return

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

    @staticmethod
    def plot_requests_per_hop_per_trial(data_file=os.path.join("data", "data_adaptive_1.json")):
        """
        Number of Requests per Hop per Trial for a specific data file.
        """
        if not os.path.exists(data_file):
            print(f"Data file {data_file} not found.")
            return

        # Load data
        with open(data_file, "r") as f:
            data = json.load(f)

        # Build records: each (trial index, hop value)
        records = []
        for trial_idx, hop_list in enumerate(data["n_hops"]):
            for hop in hop_list:
                records.append({"Trial": trial_idx, "Hops": hop})

        # Convert to DataFrame
        df = pd.DataFrame(records)
        if df.empty:
            print("No data found for requests per hop per trial.")
            return

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
