import tkinter as tk
from tkinter import ttk
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import copy
import os
import datetime


class DisplayLogger:
    def __init__(self, graph_arr, nodes_layout_pos=None, shortcut_path=None, scheme='', z=0):
        self.root = tk.Tk()
        self.root.title("Quantum Network Debugger")
        self.root.geometry("1400x900")

        self.log_folder = "logs"
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = os.path.join(self.log_folder, f"sim_log_{timestamp}_{scheme}_{z}.txt")
        self.log_file = open(self.log_filename, "w", encoding="utf-8")

        # Write Header
        self.log_file.write(f"=== Simulation Started: {timestamp} ===\n")
        self.log_file.write(f"Shortcut Path: {shortcut_path}\n")
        self.log_file.write("=" * 50 + "\n\n")

        # Simulation Data
        self.base_graph_arr = graph_arr
        self.G = nx.Graph(graph_arr)
        # Calculate fixed positions for nodes so they don't jump around
        self.pos = nodes_layout_pos if nodes_layout_pos else nx.spring_layout(self.G, seed=42)

        # Shortcut tracking
        self.shortcut_path = shortcut_path if shortcut_path else []
        self.shortcut_nodes = set(self.shortcut_path)

        # History Storage (Time Travel)
        self.history = []  # List of snapshots
        self.current_step_index = -1
        self.max_step_index = -1

        # Control Flags
        self.paused = True
        self.fast_forward = False

        self._setup_ui()

    def _setup_ui(self):
        # --- Main Layout ---
        # Split window: Left (Graph), Right (Logs/Info), Bottom (Controls)
        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # Left Frame: Graph
        self.left_frame = tk.Frame(main_pane, bg="white")
        main_pane.add(self.left_frame, minsize=700)

        # Right Frame: Info with Tabs
        self.right_frame = tk.Frame(main_pane, bg="#f0f0f0")
        main_pane.add(self.right_frame, minsize=400)

        # --- Graph Visualization (Matplotlib) ---
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- Right Panel: Info & Logs ---
        # Top: Step Info
        info_frame = tk.Frame(self.right_frame, bg="#f0f0f0")
        info_frame.pack(pady=10, fill=tk.X)

        self.lbl_step = tk.Label(info_frame, text="Step: 0", font=("Arial", 12, "bold"), bg="#f0f0f0")
        self.lbl_step.pack()

        # NEW: Request Label
        self.lbl_request = tk.Label(info_frame, text="Current Request: None", font=("Arial", 11, "bold"), fg="blue", bg="#f0f0f0")
        self.lbl_request.pack(pady=5)

        self.lbl_stats = tk.Label(info_frame, text="", font=("Arial", 10), bg="#f0f0f0", justify=tk.LEFT)
        self.lbl_stats.pack()

        # Tabbed interface for different log types
        self.notebook = ttk.Notebook(self.right_frame)
        self.notebook.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        # Tab 1: All Events
        self.tab_all = tk.Frame(self.notebook)
        self.notebook.add(self.tab_all, text="All Events")
        self.log_all = tk.Text(self.tab_all, height=30, width=50, font=("Consolas", 9))
        self.log_all.pack(fill=tk.BOTH, expand=True)

        # Tab 2: Shortcut Events
        self.tab_shortcut = tk.Frame(self.notebook)
        self.notebook.add(self.tab_shortcut, text="Shortcut Events")
        self.log_shortcut = tk.Text(self.tab_shortcut, height=30, width=50, font=("Consolas", 9))
        self.log_shortcut.pack(fill=tk.BOTH, expand=True)

        # Tab 3: Normal Routing Events
        self.tab_routing = tk.Frame(self.notebook)
        self.notebook.add(self.tab_routing, text="Normal Routing")
        self.log_routing = tk.Text(self.tab_routing, height=30, width=50, font=("Consolas", 9))
        self.log_routing.pack(fill=tk.BOTH, expand=True)

        # Tab 4: Statistics
        self.tab_stats = tk.Frame(self.notebook)
        self.notebook.add(self.tab_stats, text="Statistics")
        self.log_stats = tk.Text(self.tab_stats, height=30, width=50, font=("Consolas", 9))
        self.log_stats.pack(fill=tk.BOTH, expand=True)

        # --- Bottom Control Panel ---
        self.control_frame = tk.Frame(self.root, bg="grey")
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X)

        btn_prev = tk.Button(self.control_frame, text="<< Prev", command=self.go_back)
        btn_prev.pack(side=tk.LEFT, padx=20, pady=10)

        btn_next = tk.Button(self.control_frame, text="Next >>", command=self.go_forward)
        btn_next.pack(side=tk.LEFT, padx=5, pady=10)

        btn_run = tk.Button(self.control_frame, text="Run to End (Fast)", command=self.run_all)
        btn_run.pack(side=tk.RIGHT, padx=20, pady=10)

    def log_state(self, sim_time, nodes, requests, events_log, shortcut_active=False, current_route=None, active_request=None):
        """
        Called inside the simulation loop.
        Saves a snapshot and pauses if necessary.
        """
        # Categorize events
        shortcut_events = []
        routing_events = []
        all_events = []

        for event in events_log:
            all_events.append(event)
            if "[SHORTCUT]" in event:
                shortcut_events.append(event.replace("[SHORTCUT] ", ""))
            elif "[ROUTE]" in event:
                routing_events.append(event.replace("[ROUTE] ", ""))
            elif "[LINK]" in event:
                if self._is_shortcut_link(event):
                    shortcut_events.append(event.replace("[LINK] ", ""))
                else:
                    routing_events.append(event.replace("[LINK] ", ""))

        # Extract statistics
        stats = self._extract_statistics(nodes, shortcut_active, current_route)

        # Format Active Request String
        active_req_str = "None"
        if active_request:
            try:
                # Assuming request object has 'uid' and 'pair' attributes based on main.py
                active_req_str = f"#{active_request.uid}: Node {active_request.pair[0]} -> Node {active_request.pair[1]}"
            except AttributeError:
                active_req_str = str(active_request)

        self._write_step_to_file(sim_time, active_req_str, all_events, stats, shortcut_active)

        # Create Snapshot
        snapshot = {
            "time": sim_time,
            "nodes_state": self._extract_node_states(nodes),
            "requests": str(requests),
            "active_request_str": active_req_str,  # Store the formatted string
            "all_events": all_events,
            "shortcut_events": shortcut_events,
            "routing_events": routing_events,
            "statistics": stats,
            "shortcut_active": shortcut_active,
            "current_route": current_route if current_route else []
        }

        self.history.append(snapshot)
        self.max_step_index += 1

        # Auto-advance index
        self.current_step_index = self.max_step_index

        # Update View
        if not self.fast_forward:
            self.render_current_step()
            self.paused = True

            # BLOCKING LOOP: Wait here until user clicks Next or Run
            while self.paused:
                self.root.update()

    def _write_step_to_file(self, time, req_str, events, stats, shortcut_active):
        """Helper to format and write the current step to disk including memory and routes."""
        if not self.log_file:
            return

        separator = "-" * 60
        lines = [
            separator,
            f"TIME: {time}",
            f"Active Request: {req_str}",
            f"Shortcut Active: {shortcut_active}",
            f"Shortcut Path: {self.shortcut_path if self.shortcut_path else 'None'}",
        ]

        # Add Current Route (from history if available)
        current_route = []
        if self.history and self.current_step_index >= 0:
            current_route = self.history[self.current_step_index].get("current_route", [])
        lines.append(f"Current Route: {current_route if current_route else 'None'}")

        lines.append("")
        lines.append("--- EVENTS ---")
        if not events:
            lines.append("  (No events this step)")
        else:
            for e in events:
                lines.append(f"  {e}")

        # --- NEW: NODE MEMORY STATES ---
        lines.append("")
        lines.append("--- NODE MEMORY STATES ---")
        if self.history and self.current_step_index >= 0:
            node_states = self.history[self.current_step_index]["nodes_state"]
            for node_idx, state in node_states.items():
                if state['entanglements']:
                    links_str = ""
                    for t in state['entanglements']:
                        # t is (label, is_fixed)
                        tag = " [FIXED]" if t[1] else ""
                        links_str += f"-> Node {t[0]}{tag} | "

                    prefix = "[SC]" if state['is_shortcut_node'] else "    "
                    lines.append(f"  {prefix} Node {node_idx}: {links_str}")

        lines.append("")
        lines.append("--- STATISTICS ---")
        lines.append(f"  Total Entanglements: {stats['total_entanglements']}")
        lines.append(f"  Shortcut Links:      {stats['shortcut_entanglements']}")
        lines.append(f"  Route Links:         {stats['route_entanglements']}")
        lines.append(f"  Fixed Links:         {stats['fixed_entanglements']}")
        lines.append(f"  Memories Used:       {stats['memories_used']} / {stats['total_memories']}")
        lines.append("\n")

        self.log_file.write("\n".join(lines))
        self.log_file.flush()

    def _is_shortcut_link(self, event):
        """Check if a link creation event involves shortcut nodes"""
        if "Node" not in event:
            return False
        parts = event.split("Node")
        if len(parts) < 3:
            return False
        try:
            node1 = int(parts[1].split()[0])
            node2 = int(parts[2].split()[0])
            return node1 in self.shortcut_nodes or node2 in self.shortcut_nodes
        except:
            return False

    def _extract_statistics(self, nodes, shortcut_active, current_route):
        """Extract detailed statistics including Fixed Memory usage"""
        stats = {
            "total_entanglements": 0,
            "shortcut_entanglements": 0,
            "route_entanglements": 0,
            "fixed_entanglements": 0,  # <--- NEW
            "shortcut_active": shortcut_active,
            "memories_used": 0,
            "total_memories": 0,
            "shortcut_memories_used": 0,
            "route_memories_used": 0,
            "fixed_memories_used": 0,  # <--- NEW
            "fixed_memories_total": 0  # <--- NEW
        }

        current_route_set = set(current_route) if current_route else set()

        for node in nodes:
            stats["total_memories"] += len(node.memories)
            for mem in node.memories:
                # Track Fixed Memories
                if mem.sc_fixed:
                    stats["fixed_memories_total"] += 1
                    if mem.entangled_memory["node"] is not None:
                        stats["fixed_memories_used"] += 1
                        # We count edges as 0.5 to avoid double counting (A->B and B->A)
                        # We assume if one side is fixed, the link is a "fixed link"
                        stats["fixed_entanglements"] += 0.5

                if mem.entangled_memory["node"] is not None:
                    stats["memories_used"] += 1
                    stats["total_entanglements"] += 0.5

                    # Check Shortcut
                    if node.label in self.shortcut_nodes:
                        other_label = mem.entangled_memory["node"].label
                        if other_label in self.shortcut_nodes:
                            stats["shortcut_entanglements"] += 0.5
                            stats["shortcut_memories_used"] += 1

                    # Check Route
                    if node.label in current_route_set:
                        other_label = mem.entangled_memory["node"].label
                        if other_label in current_route_set:
                            stats["route_entanglements"] += 0.5
                            stats["route_memories_used"] += 1

        # Cast to int for display
        stats["total_entanglements"] = int(stats["total_entanglements"])
        stats["shortcut_entanglements"] = int(stats["shortcut_entanglements"])
        stats["route_entanglements"] = int(stats["route_entanglements"])
        stats["fixed_entanglements"] = int(stats["fixed_entanglements"])

        return stats

    def _extract_node_states(self, nodes):
        """Updated to pass memory flags (is_fixed) to the renderer"""
        states = {}
        for n in nodes:
            entanglements = []
            for mem in n.memories:
                if mem.entangled_memory and mem.entangled_memory["node"]:
                    target_label = mem.entangled_memory["node"].label
                    # Store tuple: (target_label, is_fixed_memory)
                    entanglements.append((target_label, mem.sc_fixed))

            states[n.label] = {
                "memories_used": len([m for m in n.memories if m.entangled_memory["node"] is not None]),
                "entanglements": entanglements,
                "is_shortcut_node": n.label in self.shortcut_nodes
            }
        return states

    def render_current_step(self):
        """Updated to draw Fixed connections in GOLD"""
        if self.current_step_index < 0 or self.current_step_index >= len(self.history):
            return

        data = self.history[self.current_step_index]
        self.ax.clear()

        # 1. Draw Base Graph
        node_colors = ['lightblue' if i in self.shortcut_nodes else 'lightgrey'
                       for i in self.G.nodes()]
        nx.draw_networkx_nodes(self.G, self.pos, ax=self.ax, node_color=node_colors, node_size=500)
        nx.draw_networkx_edges(self.G, self.pos, ax=self.ax, edge_color='grey', alpha=0.3)
        nx.draw_networkx_labels(self.G, self.pos, ax=self.ax)

        # 2. Draw Entanglements (Dynamic Overlay)
        entanglement_edges = []
        shortcut_edges = []
        route_edges = []
        fixed_edges = []  # <--- NEW LIST

        node_states = data["nodes_state"]
        current_route = set(data.get("current_route", []))

        for node_idx, state in node_states.items():
            # entanglements is now a list of (target, is_fixed)
            for target_info in state["entanglements"]:
                target_idx = target_info[0]
                is_fixed = target_info[1]

                edge = tuple(sorted((node_idx, target_idx)))

                is_shortcut = node_idx in self.shortcut_nodes and target_idx in self.shortcut_nodes
                is_route = node_idx in current_route and target_idx in current_route

                # Priority: Fixed > Shortcut > Route > Standard
                if is_fixed:
                    if edge not in fixed_edges:
                        fixed_edges.append(edge)
                elif is_shortcut:
                    if edge not in shortcut_edges:
                        shortcut_edges.append(edge)
                elif is_route:
                    if edge not in route_edges:
                        route_edges.append(edge)
                else:
                    if edge not in entanglement_edges:
                        entanglement_edges.append(edge)

        # Draw edges
        if fixed_edges:
            # GOLD for fixed connections
            nx.draw_networkx_edges(self.G, self.pos, edgelist=fixed_edges,
                                   ax=self.ax, edge_color='gold', style='solid', width=4, alpha=0.9)
        if shortcut_edges:
            nx.draw_networkx_edges(self.G, self.pos, edgelist=shortcut_edges,
                                   ax=self.ax, edge_color='blue', style='solid', width=3, alpha=0.8)
        if route_edges:
            nx.draw_networkx_edges(self.G, self.pos, edgelist=route_edges,
                                   ax=self.ax, edge_color='green', style='dashed', width=2.5, alpha=0.7)
        if entanglement_edges:
            nx.draw_networkx_edges(self.G, self.pos, edgelist=entanglement_edges,
                                   ax=self.ax, edge_color='red', style='dotted', width=2, alpha=0.6)

        title = f"Time: {data['time']}"
        if data.get('shortcut_active'):
            title += " [SHORTCUT ACTIVE]"
        self.ax.set_title(title, fontweight='bold')
        self.canvas.draw()

        # Update Labels and Logs
        req_str = data.get("active_request_str", "None")
        self.lbl_request.config(text=f"Current Request: {req_str}")
        self.lbl_step.config(text=f"Time: {data['time']} | Step: {self.current_step_index}/{self.max_step_index}")

        # Update Stats Label
        stats = data["statistics"]
        stats_text = f"Shortcut Active: {'YES' if stats['shortcut_active'] else 'NO'}\n"
        stats_text += f"Fixed Links: {stats['fixed_entanglements']}\n"  # <--- ADDED
        stats_text += f"Total Entanglements: {stats['total_entanglements']}"
        self.lbl_stats.config(text=stats_text)

        # Update Stats Tab
        self.log_stats.delete(1.0, tk.END)
        self.log_stats.insert(tk.END, f"=== STATISTICS @ Time {data['time']} ===\n\n")
        self.log_stats.insert(tk.END, f"Active Request: {req_str}\n")
        self.log_stats.insert(tk.END, f"Shortcut Active: {stats['shortcut_active']}\n\n")

        self.log_stats.insert(tk.END, f"--- Special ---\n")
        self.log_stats.insert(tk.END, f"Fixed Links: {stats['fixed_entanglements']}\n")
        self.log_stats.insert(tk.END,
                              f"Fixed Memos: {stats['fixed_memories_used']} / {stats['fixed_memories_total']}\n\n")

        self.log_stats.insert(tk.END, f"--- Entanglements ---\n")
        self.log_stats.insert(tk.END, f"Total: {stats['total_entanglements']}\n")
        self.log_stats.insert(tk.END, f"Shortcut: {stats['shortcut_entanglements']}\n")
        self.log_stats.insert(tk.END, f"Route: {stats['route_entanglements']}\n")

        self.log_stats.insert(tk.END, f"\n--- Memory Usage ---\n")
        self.log_stats.insert(tk.END, f"Used: {stats['memories_used']} / {stats['total_memories']}\n")
        self.log_stats.insert(tk.END, f"Utilization: {stats['memories_used'] / stats['total_memories'] * 100:.1f}%\n")

        # Update Node States in Log (to show [FIXED])
        self.log_all.delete(1.0, tk.END)
        self.log_all.insert(tk.END, f"=== TIME {data['time']} ===\n")
        self.log_all.insert(tk.END, f"REQ: {req_str}\n\n")
        for event in data["all_events"]:
            self.log_all.insert(tk.END, f"{event}\n")

        self.log_all.insert(tk.END, "\n--- NODE STATES ---\n")
        for node_idx, state in node_states.items():
            if state['entanglements']:
                links_str = ""
                for t in state['entanglements']:
                    # t is (label, is_fixed)
                    tag = "[FIXED]," if t[1] else ","
                    links_str += f"{t[0]}{tag} "

                prefix = "[SC]" if state['is_shortcut_node'] else "   "
                self.log_all.insert(tk.END, f"{prefix} Node {node_idx}: {links_str}\n")

    # --- Button Commands ---
    def go_forward(self):
        if self.current_step_index < self.max_step_index:
            self.current_step_index += 1
            self.render_current_step()
        else:
            self.paused = False

    def go_back(self):
        if self.current_step_index > 0:
            self.current_step_index -= 1
            self.render_current_step()

    def run_all(self):
        self.fast_forward = True
        self.paused = False