import os
import matplotlib.pyplot as plt
import numpy as np
import datetime

class Plots:
    def __init__(self):
        self.output_dir = 'stats'

    def plot_timing_schemes(self, data):
        """
        Create timing comparison plots for each scheme.

        Parameters
        ----------
        data : dict
            Nested dictionary structured as:
            data[scheme][shortcut_flag][trial] = [(request, time), ...]
            where shortcut_flag is 0 (no shortcut) or 1 (with shortcut).
        output_dir : str
            Directory where plots will be saved.
        """

        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        # Modern, accessible color scheme
        colors = {
            'no_shortcut': '#1F77B4',   # Matplotlib default blue
            'shortcut': '#FF7F0E',   # Matplotlib default orange
        }

        for scheme_name, scheme_data in data.items():
            fig, axes = plt.subplots(2, 5, figsize=(24, 10), sharex=False, sharey=False)
            fig.patch.set_facecolor('#FAFAFA')  # Light background
            fig.suptitle(
                f"Timing Comparison – {scheme_name} (Shortcut vs No Shortcut)",
                fontsize=16,
                fontweight='bold',
                y=0.98
            )

            for trial_idx in range(10):
                row = trial_idx // 5
                col = trial_idx % 5
                ax = axes[row, col]

                # Styling for each subplot
                ax.set_facecolor('#FFFFFF')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#CCCCCC')
                ax.spines['bottom'].set_color('#CCCCCC')

                # Grid lines for better readability
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='#CCCCCC')
                ax.set_axisbelow(True)  # Grid behind data

                has_data = False

                # Plot BOTH shortcut and no-shortcut in the same subplot
                for shortcut_flag in [0, 1]:
                    trials = scheme_data.get(shortcut_flag, {})
                    trial_data = trials.get(trial_idx, [])

                    if not trial_data:
                        continue

                    has_data = True
                    color = colors['shortcut'] if shortcut_flag else colors['no_shortcut']
                    label = "With Shortcut" if shortcut_flag else "No Shortcut"

                    x = [req for req, _ in trial_data]
                    y = [t for _, t in trial_data]

                    # Plot with markers and improved styling
                    ax.plot(x, y, color=color, linewidth=2.5, alpha=0.85,
                            marker='o', markersize=5, markerfacecolor=color,
                            markeredgewidth=0, label=label)

                if not has_data:
                    ax.set_visible(False)
                    continue

                # Title and labels
                ax.set_title(f"Trial {trial_idx}", fontsize=12, fontweight='bold', pad=10)
                ax.set_xlabel("Request Index", fontsize=10, color='#333333')
                ax.set_ylabel("Completion Time", fontsize=10, color='#333333')

                # Tick styling
                ax.tick_params(colors='#666666', labelsize=9)

                # Legend in each subplot
                ax.legend(loc='best', frameon=True, fancybox=False,
                          edgecolor='#CCCCCC', framealpha=0.95, fontsize=9)

            # Add horizontal separator line between rows
            fig.add_artist(plt.Line2D([0.05, 0.95], [0.5, 0.5],
                                      transform=fig.transFigure,
                                      color='#DDDDDD', linewidth=2))

            plt.tight_layout(rect=[0, 0.02, 1, 0.96])

            filename = f"timing_{timestamp}_{scheme_name}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, facecolor='#FAFAFA', bbox_inches='tight')
            plt.close(fig)

    def plot_win_percentage(self, data, x_label_interval=10):
        """
        Create stacked bar plots showing win percentage for each scheme.

        For each request index, compares completion times across all trials
        to determine which condition (shortcut vs no shortcut) was faster.

        Parameters
        ----------
        data : dict
            Nested dictionary structured as:
            data[scheme][shortcut_flag][trial] = [(request, time), ...]
            where shortcut_flag is 0 (no shortcut) or 1 (with shortcut).
        x_label_interval : int
            Show x-axis label every N requests (default: 10)
        """

        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        # Color scheme - more distinct shortcut color
        colors = {
            'no_shortcut': '#1F77B4',   # Matplotlib default blue
            'shortcut': '#FF7F0E',   # Matplotlib default orange
            'tie': '#B0B0B0'  # Gray for ties
        }

        for scheme_name, scheme_data in data.items():
            # Collect all request indices across all trials
            all_request_indices = set()

            for shortcut_flag in [0, 1]:
                trials = scheme_data.get(shortcut_flag, {})
                for trial_idx, trial_data in trials.items():
                    for req_idx, _ in trial_data:
                        all_request_indices.add(req_idx)

            if not all_request_indices:
                continue

            request_indices = sorted(all_request_indices)

            # For each request index, count wins across all trials
            win_counts = {req_idx: {'no_shortcut': 0, 'shortcut': 0, 'tie': 0}
                          for req_idx in request_indices}

            # Get number of trials
            num_trials = max(
                len(scheme_data.get(0, {})),
                len(scheme_data.get(1, {}))
            )

            # Compare each request across all trials
            for req_idx in request_indices:
                for trial_idx in range(num_trials):
                    # Get times for this request in this trial
                    no_shortcut_trials = scheme_data.get(0, {})
                    shortcut_trials = scheme_data.get(1, {})

                    no_shortcut_data = no_shortcut_trials.get(trial_idx, [])
                    shortcut_data = shortcut_trials.get(trial_idx, [])

                    # Find the time for this specific request index
                    no_shortcut_time = None
                    shortcut_time = None

                    for req, time in no_shortcut_data:
                        if req == req_idx:
                            no_shortcut_time = time
                            break

                    for req, time in shortcut_data:
                        if req == req_idx:
                            shortcut_time = time
                            break

                    # Compare times
                    if no_shortcut_time is not None and shortcut_time is not None:
                        if no_shortcut_time < shortcut_time:
                            win_counts[req_idx]['no_shortcut'] += 1
                        elif shortcut_time < no_shortcut_time:
                            win_counts[req_idx]['shortcut'] += 1
                        else:
                            win_counts[req_idx]['tie'] += 1

            # Convert counts to percentages
            win_percentages = {req_idx: {} for req_idx in request_indices}
            for req_idx in request_indices:
                total = sum(win_counts[req_idx].values())
                if total > 0:
                    win_percentages[req_idx]['no_shortcut'] = (win_counts[req_idx]['no_shortcut'] / total) * 100
                    win_percentages[req_idx]['shortcut'] = (win_counts[req_idx]['shortcut'] / total) * 100
                    win_percentages[req_idx]['tie'] = (win_counts[req_idx]['tie'] / total) * 100
                else:
                    win_percentages[req_idx] = {'no_shortcut': 0, 'shortcut': 0, 'tie': 0}

            # Prepare data for stacked bar plot
            no_shortcut_pct = [win_percentages[req_idx]['no_shortcut'] for req_idx in request_indices]
            tie_pct = [win_percentages[req_idx]['tie'] for req_idx in request_indices]
            shortcut_pct = [win_percentages[req_idx]['shortcut'] for req_idx in request_indices]

            # Create the plot
            fig, ax = plt.subplots(figsize=(20, 8))
            fig.patch.set_facecolor('#FAFAFA')

            x_pos = np.arange(len(request_indices))

            # Create stacked bars
            p1 = ax.bar(x_pos, no_shortcut_pct, color=colors['no_shortcut'],
                        alpha=0.85, label='No Shortcut Wins', edgecolor='white', linewidth=0.5)
            p2 = ax.bar(x_pos, tie_pct, bottom=no_shortcut_pct, color=colors['tie'],
                        alpha=0.85, label='Ties', edgecolor='white', linewidth=0.5)
            p3 = ax.bar(x_pos, shortcut_pct,
                        bottom=np.array(no_shortcut_pct) + np.array(tie_pct),
                        color=colors['shortcut'], alpha=0.85, label='Shortcut Wins',
                        edgecolor='white', linewidth=0.5)

            # Add 50% reference line
            ax.axhline(y=50, color='#333333', linestyle='--', linewidth=1.5,
                       alpha=0.7, label='50% Reference', zorder=10)

            # Styling
            ax.set_facecolor('#FFFFFF')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#CCCCCC')
            ax.spines['bottom'].set_color('#CCCCCC')

            # Grid
            ax.yaxis.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='#CCCCCC')
            ax.set_axisbelow(True)

            # Labels and title
            ax.set_xlabel('Request Index', fontsize=12, fontweight='bold', color='#333333')
            ax.set_ylabel('Win Percentage (%)', fontsize=12, fontweight='bold', color='#333333')
            ax.set_title(f'Win Percentage Comparison – {scheme_name}\n(Across {num_trials} Trials)',
                         fontsize=14, fontweight='bold', pad=20)

            # X-axis labels - show only every x_label_interval-th label
            ax.set_xticks(x_pos)
            x_labels = [str(req_idx) if i % x_label_interval == 0 else ''
                        for i, req_idx in enumerate(request_indices)]
            ax.set_xticklabels(x_labels, fontsize=9)
            ax.set_ylim(0, 100)

            # Tick styling
            ax.tick_params(colors='#666666', labelsize=10)

            # Legend
            ax.legend(loc='upper right', frameon=True, fancybox=False,
                      edgecolor='#CCCCCC', framealpha=0.95, fontsize=11)

            plt.tight_layout()

            filename = f"win_percentage_{timestamp}_{scheme_name}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, facecolor='#FAFAFA', bbox_inches='tight')
            plt.close(fig)

    def plot_shortcut_usage(self, shortcut_data, total_requests_data):
        """
        Create plots showing shortcut usage statistics for each scheme.

        Parameters
        ----------
        shortcut_data : dict
            Dictionary structured as:
            shortcut_data[scheme][trial] = number of shortcuts taken
        total_requests_data : dict
            Dictionary structured as:
            total_requests_data[scheme][trial] = total number of requests served
        """

        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        # Modern, accessible color scheme
        colors = {
            'powerlaw': '#1F77B4',  # Blue
            'uniform': '#FF7F0E',  # Orange
            'adaptive': '#2CA02C',  # Green
        }

        # Create a figure with subplots for each scheme
        num_schemes = len(shortcut_data)
        fig, axes = plt.subplots(1, num_schemes, figsize=(8 * num_schemes, 6))

        # Handle case where there's only one scheme
        if num_schemes == 1:
            axes = [axes]

        fig.patch.set_facecolor('#FAFAFA')
        fig.suptitle(
            "Shortcut Usage Analysis Across Schemes and Trials",
            fontsize=16,
            fontweight='bold',
            y=0.98
        )

        for idx, (scheme_name, scheme_shortcuts) in enumerate(shortcut_data.items()):
            ax = axes[idx]

            # Get data for this scheme
            trials = sorted(scheme_shortcuts.keys())
            shortcuts_taken = [scheme_shortcuts[t] for t in trials]
            total_requests = [total_requests_data[scheme_name][t] for t in trials]

            # Calculate percentages
            percentages = [(s / t * 100) if t > 0 else 0
                           for s, t in zip(shortcuts_taken, total_requests)]

            # Styling for each subplot
            ax.set_facecolor('#FFFFFF')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#CCCCCC')
            ax.spines['bottom'].set_color('#CCCCCC')

            # Grid lines
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5,
                    color='#CCCCCC', axis='y')
            ax.set_axisbelow(True)

            # Create bar plot
            x_pos = np.arange(len(trials))
            bars = ax.bar(x_pos, shortcuts_taken, color=colors.get(scheme_name, '#1F77B4'),
                          alpha=0.85, edgecolor='white', linewidth=1)

            # Add percentage labels on top of bars
            for i, (bar, pct) in enumerate(zip(bars, percentages)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{pct:.1f}%',
                        ha='center', va='bottom', fontsize=9,
                        fontweight='bold', color='#333333')

            # Labels and title
            ax.set_xlabel('Trial', fontsize=11, fontweight='bold', color='#333333')
            ax.set_ylabel('Number of Shortcuts Taken', fontsize=11,
                          fontweight='bold', color='#333333')
            ax.set_title(f'{scheme_name.capitalize()}', fontsize=13,
                         fontweight='bold', pad=15)

            # X-axis
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'T{t}' for t in trials], fontsize=10)

            # Tick styling
            ax.tick_params(colors='#666666', labelsize=10)

            # Add statistics text box
            mean_shortcuts = np.mean(shortcuts_taken)
            mean_percentage = np.mean(percentages)
            stats_text = f'Mean: {mean_shortcuts:.1f} ({mean_percentage:.1f}%)'
            ax.text(0.98, 0.98, stats_text,
                    transform=ax.transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='#FFFFFF',
                              edgecolor='#CCCCCC', alpha=0.9))

        plt.tight_layout(rect=[0, 0.02, 1, 0.96])

        filename = f"shortcut_usage_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, facecolor='#FAFAFA', bbox_inches='tight')
        plt.close(fig)

        print(f"Shortcut usage plot saved to: {filepath}")

        # Create a second plot: comparison across schemes
        self._plot_shortcut_comparison(shortcut_data, total_requests_data, timestamp)

    def _plot_shortcut_comparison(self, shortcut_data, total_requests_data, timestamp):
        """
        Create a grouped bar plot comparing shortcut usage across schemes.
        """

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor('#FAFAFA')
        fig.suptitle(
            "Shortcut Usage Comparison Across Schemes",
            fontsize=16,
            fontweight='bold',
            y=0.98
        )

        schemes = list(shortcut_data.keys())
        colors = {
            'powerlaw': '#1F77B4',
            'uniform': '#FF7F0E',
            'adaptive': '#2CA02C',
        }

        # Plot 1: Average shortcuts per scheme
        ax = ax1
        ax.set_facecolor('#FFFFFF')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5,
                color='#CCCCCC', axis='y')
        ax.set_axisbelow(True)

        # Calculate averages
        avg_shortcuts = []
        avg_percentages = []
        for scheme in schemes:
            shortcuts = list(shortcut_data[scheme].values())
            totals = list(total_requests_data[scheme].values())
            avg_shortcuts.append(np.mean(shortcuts))
            avg_pct = np.mean([s / t * 100 for s, t in zip(shortcuts, totals)])
            avg_percentages.append(avg_pct)

        x_pos = np.arange(len(schemes))
        bars = ax.bar(x_pos, avg_shortcuts,
                      color=[colors.get(s, '#1F77B4') for s in schemes],
                      alpha=0.85, edgecolor='white', linewidth=1)

        # Add percentage labels
        for bar, pct in zip(bars, avg_percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{pct:.1f}%',
                    ha='center', va='bottom', fontsize=10,
                    fontweight='bold', color='#333333')

        ax.set_xlabel('Scheme', fontsize=11, fontweight='bold', color='#333333')
        ax.set_ylabel('Average Shortcuts Taken', fontsize=11,
                      fontweight='bold', color='#333333')
        ax.set_title('Average Shortcut Usage by Scheme', fontsize=13,
                     fontweight='bold', pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([s.capitalize() for s in schemes], fontsize=11)
        ax.tick_params(colors='#666666', labelsize=10)

        # Plot 2: Box plot showing distribution
        ax = ax2
        ax.set_facecolor('#FFFFFF')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5,
                color='#CCCCCC', axis='y')
        ax.set_axisbelow(True)

        # Prepare data for box plot
        box_data = []
        for scheme in schemes:
            shortcuts = list(shortcut_data[scheme].values())
            totals = list(total_requests_data[scheme].values())
            percentages = [s / t * 100 for s, t in zip(shortcuts, totals)]
            box_data.append(percentages)

        bp = ax.boxplot(box_data, labels=[s.capitalize() for s in schemes],
                        patch_artist=True, widths=0.6,
                        boxprops=dict(linewidth=1.5, edgecolor='#666666'),
                        whiskerprops=dict(linewidth=1.5, color='#666666'),
                        capprops=dict(linewidth=1.5, color='#666666'),
                        medianprops=dict(linewidth=2, color='#D62728'))

        # Color the boxes
        for patch, scheme in zip(bp['boxes'], schemes):
            patch.set_facecolor(colors.get(scheme, '#1F77B4'))
            patch.set_alpha(0.85)

        ax.set_xlabel('Scheme', fontsize=11, fontweight='bold', color='#333333')
        ax.set_ylabel('Shortcut Usage (%)', fontsize=11,
                      fontweight='bold', color='#333333')
        ax.set_title('Distribution of Shortcut Usage', fontsize=13,
                     fontweight='bold', pad=15)
        ax.tick_params(colors='#666666', labelsize=10)

        plt.tight_layout(rect=[0, 0.02, 1, 0.96])

        filename = f"shortcut_comparison_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, facecolor='#FAFAFA', bbox_inches='tight')
        plt.close(fig)

        print(f"Shortcut comparison plot saved to: {filepath}")