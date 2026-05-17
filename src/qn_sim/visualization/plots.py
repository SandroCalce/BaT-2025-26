import os
import matplotlib.pyplot as plt
import numpy as np
import datetime

class Plots:
    def __init__(self, run_dir=None, data_dir=None):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_dir is None:
            run_dir = os.path.join('stats', timestamp)
        if data_dir is None:
            data_dir = os.path.join('data', timestamp)
        self.output_dir = run_dir
        self.data_dir = data_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

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

                    x = [req for req, _, _ in trial_data]
                    y = [t for _, t, _ in trial_data]

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

            filename = f"timing_{scheme_name}.png"
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
                    for req_idx, _, _ in trial_data:
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

                    for req, time, _ in no_shortcut_data:
                        if req == req_idx:
                            no_shortcut_time = time
                            break

                    for req, time, _ in shortcut_data:
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

            filename = f"win_percentage_{scheme_name}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, facecolor='#FAFAFA', bbox_inches='tight')
            plt.close(fig)

    def plot_service_win_percentage(self, data, x_label_interval=10):
        """
        Create stacked bar plots showing service time win percentage for each scheme.

        Compares 'serve_time' (active duration) instead of absolute completion time.

        Parameters
        ----------
        data : dict
            Nested dictionary structured as:
            data[scheme][shortcut_flag][trial] = [(request, end_time, serve_time), ...]
        x_label_interval : int
            Show x-axis label every N requests.
        """

        colors = {
            'no_shortcut': '#1F77B4',
            'shortcut': '#FF7F0E',
            'tie': '#B0B0B0'
        }

        for scheme_name, scheme_data in data.items():
            all_request_indices = set()
            for shortcut_flag in [0, 1]:
                trials = scheme_data.get(shortcut_flag, {})
                for trial_idx, trial_data in trials.items():
                    for req_idx, _, _ in trial_data:
                        all_request_indices.add(req_idx)

            if not all_request_indices:
                continue

            request_indices = sorted(all_request_indices)
            win_counts = {req_idx: {'no_shortcut': 0, 'shortcut': 0, 'tie': 0}
                          for req_idx in request_indices}

            num_trials = max(len(scheme_data.get(0, {})), len(scheme_data.get(1, {})))

            for req_idx in request_indices:
                for trial_idx in range(num_trials):
                    no_shortcut_trials = scheme_data.get(0, {})
                    shortcut_trials = scheme_data.get(1, {})

                    no_shortcut_data = no_shortcut_trials.get(trial_idx, [])
                    shortcut_data = shortcut_trials.get(trial_idx, [])

                    no_shortcut_serve_time = None
                    shortcut_serve_time = None

                    for req, _, stime in no_shortcut_data:
                        if req == req_idx:
                            no_shortcut_serve_time = stime
                            break

                    for req, _, stime in shortcut_data:
                        if req == req_idx:
                            shortcut_serve_time = stime
                            break

                    if no_shortcut_serve_time is not None and shortcut_serve_time is not None:
                        if no_shortcut_serve_time < shortcut_serve_time:
                            win_counts[req_idx]['no_shortcut'] += 1
                        elif shortcut_serve_time < no_shortcut_serve_time:
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

            no_shortcut_pct = [win_percentages[req_idx]['no_shortcut'] for req_idx in request_indices]
            tie_pct = [win_percentages[req_idx]['tie'] for req_idx in request_indices]
            shortcut_pct = [win_percentages[req_idx]['shortcut'] for req_idx in request_indices]

            fig, ax = plt.subplots(figsize=(20, 8))
            fig.patch.set_facecolor('#FAFAFA')
            x_pos = np.arange(len(request_indices))

            ax.bar(x_pos, no_shortcut_pct, color=colors['no_shortcut'], alpha=0.85, label='No Shortcut Faster (Service)', edgecolor='white', linewidth=0.5)
            ax.bar(x_pos, tie_pct, bottom=no_shortcut_pct, color=colors['tie'], alpha=0.85, label='Ties', edgecolor='white', linewidth=0.5)
            ax.bar(x_pos, shortcut_pct, bottom=np.array(no_shortcut_pct) + np.array(tie_pct), color=colors['shortcut'], alpha=0.85, label='Shortcut Faster (Service)', edgecolor='white', linewidth=0.5)

            ax.axhline(y=50, color='#333333', linestyle='--', linewidth=1.5, alpha=0.7, label='50% Reference', zorder=10)

            ax.set_facecolor('#FFFFFF')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlabel('Request Index', fontsize=12, fontweight='bold', color='#333333')
            ax.set_ylabel('Service Time Win Percentage (%)', fontsize=12, fontweight='bold', color='#333333')
            ax.set_title(f'Service Time Efficiency Comparison – {scheme_name}\n(Shortcut vs No Shortcut)', fontsize=14, fontweight='bold', pad=20)

            ax.set_xticks(x_pos)
            x_labels = [str(req_idx) if i % x_label_interval == 0 else '' for i, req_idx in enumerate(request_indices)]
            ax.set_xticklabels(x_labels, fontsize=9)
            ax.set_ylim(0, 100)
            ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='#CCCCCC', framealpha=0.95, fontsize=11)

            plt.tight_layout()
            filename = f"service_win_percentage_{scheme_name}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, facecolor='#FAFAFA', bbox_inches='tight')
            plt.close(fig)

    def plot_serve_time_band(self, data):
        """
        Aggregate serve_time across trials per request index and plot a
        comparison of no-shortcut vs shortcut for each scheme.

        Each line shows the median across trials; two shaded bands are drawn
        per condition: an inner darker band for the 95% CI of the mean, and
        an outer lighter band for the 5-95 percentile range of the data.

        Parameters
        ----------
        data : dict
            data[scheme][shortcut_flag][trial] = [(uid, completion_time, serve_time), ...]
        """

        colors = {0: '#1F77B4', 1: '#FF7F0E'}
        labels = {0: 'No Shortcut', 1: 'Shortcut'}

        for scheme_name, scheme_data in data.items():
            # serve_times[z][req_idx] = [serve_time across trials]
            serve_times = {0: {}, 1: {}}
            for z in (0, 1):
                trials = scheme_data.get(z, {})
                for trial_data in trials.values():
                    for req_idx, _, stime in trial_data:
                        serve_times[z].setdefault(req_idx, []).append(stime)

            if not serve_times[0] and not serve_times[1]:
                continue

            fig, ax = plt.subplots(figsize=(14, 6))
            fig.patch.set_facecolor('#FAFAFA')

            for z in (0, 1):
                if not serve_times[z]:
                    continue

                req_indices = sorted(serve_times[z].keys())
                xs, medians, p5, p95, ci_lo, ci_hi = [], [], [], [], [], []
                for r in req_indices:
                    vals = np.array(serve_times[z][r])
                    if vals.size == 0:
                        continue
                    xs.append(r)
                    medians.append(np.median(vals))
                    p5.append(np.percentile(vals, 5))
                    p95.append(np.percentile(vals, 95))
                    mean = float(np.mean(vals))
                    sem = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
                    ci_lo.append(mean - 1.96 * sem)
                    ci_hi.append(mean + 1.96 * sem)

                if not xs:
                    continue

                xs = np.array(xs)
                color = colors[z]

                # Outer band: 5-95 percentile (real spread of the data)
                ax.fill_between(xs, p5, p95, color=color, alpha=0.12,
                                linewidth=0, label=f'{labels[z]} 5–95 percentile')
                # Inner band: 95% CI of the mean (uncertainty in the estimate)
                ax.fill_between(xs, ci_lo, ci_hi, color=color, alpha=0.30,
                                linewidth=0, label=f'{labels[z]} 95% CI of mean')
                # Median line
                ax.plot(xs, medians, color=color, linewidth=2.2,
                        label=f'{labels[z]} median', zorder=10)

            ax.set_facecolor('#FFFFFF')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#CCCCCC')
            ax.spines['bottom'].set_color('#CCCCCC')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='#CCCCCC')
            ax.set_axisbelow(True)

            ax.set_xlabel('Request Index', fontsize=12, fontweight='bold', color='#333333')
            ax.set_ylabel('Service Time', fontsize=12, fontweight='bold', color='#333333')
            ax.set_title(
                f'Service Time per Request – {scheme_name}\n'
                f'Median (line), 95% CI of mean (inner band), 5–95 percentile (outer band)',
                fontsize=13, fontweight='bold', pad=15
            )
            ax.tick_params(colors='#666666', labelsize=10)
            ax.legend(loc='upper left', frameon=True, fancybox=False,
                      edgecolor='#CCCCCC', framealpha=0.95, fontsize=10, ncol=2)

            plt.tight_layout()

            filename = f"serve_time_band_{scheme_name}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, facecolor='#FAFAFA', bbox_inches='tight')
            plt.close(fig)

            print(f"Serve-time band plot saved to: {filepath}")

    def plot_completion_time_band(self, data):
        """
        Aggregate completion time (absolute sim_time) across trials per
        request index and plot a comparison of no-shortcut vs shortcut for
        each scheme.

        Each line shows the median across trials; two shaded bands are drawn
        per condition: an inner darker band for the 95% CI of the mean, and
        an outer lighter band for the 5-95 percentile range of the data.

        Note: completion time grows monotonically with request index. The
        gap between the two lines is the cumulative wall-clock benefit of
        the shortcut.

        Parameters
        ----------
        data : dict
            data[scheme][shortcut_flag][trial] = [(uid, completion_time, serve_time), ...]
        """

        colors = {0: '#1F77B4', 1: '#FF7F0E'}
        labels = {0: 'No Shortcut', 1: 'Shortcut'}

        for scheme_name, scheme_data in data.items():
            # completion_times[z][req_idx] = [sim_time across trials]
            completion_times = {0: {}, 1: {}}
            for z in (0, 1):
                trials = scheme_data.get(z, {})
                for trial_data in trials.values():
                    for req_idx, ctime, _ in trial_data:
                        completion_times[z].setdefault(req_idx, []).append(ctime)

            if not completion_times[0] and not completion_times[1]:
                continue

            fig, ax = plt.subplots(figsize=(14, 6))
            fig.patch.set_facecolor('#FAFAFA')

            for z in (0, 1):
                if not completion_times[z]:
                    continue

                req_indices = sorted(completion_times[z].keys())
                xs, medians, p5, p95, ci_lo, ci_hi = [], [], [], [], [], []
                for r in req_indices:
                    vals = np.array(completion_times[z][r])
                    if vals.size == 0:
                        continue
                    xs.append(r)
                    medians.append(np.median(vals))
                    p5.append(np.percentile(vals, 5))
                    p95.append(np.percentile(vals, 95))
                    mean = float(np.mean(vals))
                    sem = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
                    ci_lo.append(mean - 1.96 * sem)
                    ci_hi.append(mean + 1.96 * sem)

                if not xs:
                    continue

                xs = np.array(xs)
                color = colors[z]

                ax.fill_between(xs, p5, p95, color=color, alpha=0.12,
                                linewidth=0, label=f'{labels[z]} 5–95 percentile')
                ax.fill_between(xs, ci_lo, ci_hi, color=color, alpha=0.30,
                                linewidth=0, label=f'{labels[z]} 95% CI of mean')
                ax.plot(xs, medians, color=color, linewidth=2.2,
                        label=f'{labels[z]} median', zorder=10)

            ax.set_facecolor('#FFFFFF')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#CCCCCC')
            ax.spines['bottom'].set_color('#CCCCCC')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='#CCCCCC')
            ax.set_axisbelow(True)

            ax.set_xlabel('Request Index', fontsize=12, fontweight='bold', color='#333333')
            ax.set_ylabel('Completion Time (sim_time)', fontsize=12, fontweight='bold', color='#333333')
            ax.set_title(
                f'Completion Time per Request – {scheme_name}\n'
                f'Median (line), 95% CI of mean (inner band), 5–95 percentile (outer band)',
                fontsize=13, fontweight='bold', pad=15
            )
            ax.tick_params(colors='#666666', labelsize=10)
            ax.legend(loc='upper left', frameon=True, fancybox=False,
                      edgecolor='#CCCCCC', framealpha=0.95, fontsize=10, ncol=2)

            plt.tight_layout()

            filename = f"completion_time_band_{scheme_name}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, facecolor='#FAFAFA', bbox_inches='tight')
            plt.close(fig)

            print(f"Completion-time band plot saved to: {filepath}")

    def plot_percentage_improvement(self, data, smoothing_window=None):
        """
        For each scheme, plot the percentage improvement of the shortcut
        variant over the no-shortcut baseline, paired per trial (CRN).

        Two panels per scheme:
          - Completion time: (no_sc - sc) / no_sc * 100  per request index.
            Because completion time is cumulative sim_time, this is the
            running wall-clock savings up to request X.
          - Service time: same formula on serve_time. This is the per-request
            latency improvement.

        Median across trials is drawn as a line; the 95% CI of the mean is
        a shaded band. A light rolling mean is applied for readability when
        request counts are large.

        Parameters
        ----------
        data : dict
            data[scheme][shortcut_flag][trial] = [(uid, completion_time, serve_time), ...]
        smoothing_window : int or None
            Rolling-mean window size. If None, auto-pick max(1, num_requests // 200).
        """

        for scheme_name, scheme_data in data.items():
            no_trials = scheme_data.get(0, {})
            sc_trials = scheme_data.get(1, {})
            if not no_trials or not sc_trials:
                continue

            ctime_pct = {}
            stime_pct = {}

            shared_trials = sorted(set(no_trials.keys()) & set(sc_trials.keys()))
            for trial_idx in shared_trials:
                no_data = no_trials[trial_idx]
                sc_data = sc_trials[trial_idx]
                if not no_data or not sc_data:
                    continue

                no_by_uid = {uid: (ct, st) for uid, ct, st in no_data}
                sc_by_uid = {uid: (ct, st) for uid, ct, st in sc_data}

                for uid in set(no_by_uid) & set(sc_by_uid):
                    no_ct, no_st = no_by_uid[uid]
                    sc_ct, sc_st = sc_by_uid[uid]
                    if no_ct > 0:
                        ctime_pct.setdefault(uid, []).append((no_ct - sc_ct) / no_ct * 100)
                    if no_st > 0:
                        stime_pct.setdefault(uid, []).append((no_st - sc_st) / no_st * 100)

            if not ctime_pct and not stime_pct:
                continue

            def _aggregate(pct_dict):
                uids = sorted(pct_dict.keys())
                xs, medians, ci_lo, ci_hi = [], [], [], []
                for u in uids:
                    vals = np.array(pct_dict[u])
                    if vals.size == 0:
                        continue
                    xs.append(u)
                    medians.append(float(np.median(vals)))
                    mean = float(np.mean(vals))
                    sem = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
                    ci_lo.append(mean - 1.96 * sem)
                    ci_hi.append(mean + 1.96 * sem)
                return np.array(xs), np.array(medians), np.array(ci_lo), np.array(ci_hi)

            def _smooth(arr, win):
                if win <= 1 or arr.size == 0:
                    return arr
                kernel = np.ones(win) / win
                return np.convolve(arr, kernel, mode='same')

            fig, (ax_c, ax_s) = plt.subplots(1, 2, figsize=(18, 6))
            fig.patch.set_facecolor('#FAFAFA')
            fig.suptitle(
                f"Shortcut Speedup vs No-Shortcut – {scheme_name}\n"
                f"Paired per trial; positive % means shortcut is faster",
                fontsize=14, fontweight='bold', y=0.99
            )

            color = '#FF7F0E'
            panels = [
                (ax_c, ctime_pct, 'Cumulative Completion-Time Speedup',
                 'Speedup (%) — (no_sc − sc) / no_sc'),
                (ax_s, stime_pct, 'Per-Request Service-Time Speedup',
                 'Speedup (%) — (no_sc − sc) / no_sc'),
            ]

            for ax, pct_dict, title, ylabel in panels:
                ax.set_facecolor('#FFFFFF')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#CCCCCC')
                ax.spines['bottom'].set_color('#CCCCCC')
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='#CCCCCC')
                ax.set_axisbelow(True)
                ax.axhline(y=0, color='#333333', linestyle='--', linewidth=1.2,
                           alpha=0.7, zorder=5)

                xs, medians, ci_lo, ci_hi = _aggregate(pct_dict)
                if xs.size == 0:
                    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
                    continue

                win = smoothing_window if smoothing_window is not None else max(1, xs.size // 200)
                medians_s = _smooth(medians, win)
                ci_lo_s = _smooth(ci_lo, win)
                ci_hi_s = _smooth(ci_hi, win)

                ax.fill_between(xs, ci_lo_s, ci_hi_s, color=color, alpha=0.25,
                                linewidth=0, label='95% CI of mean')
                ax.plot(xs, medians_s, color=color, linewidth=2.2,
                        label=f'Median (window={win})', zorder=10)

                final_med = float(medians_s[-1])
                ax.annotate(
                    f'{final_med:+.1f}% @ req {int(xs[-1])}',
                    xy=(xs[-1], final_med),
                    xytext=(-10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold', color='#333333',
                    ha='right',
                    bbox=dict(boxstyle='round', facecolor='#FFFFFF',
                              edgecolor='#CCCCCC', alpha=0.9)
                )

                ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
                ax.set_xlabel('Request Index', fontsize=11, fontweight='bold', color='#333333')
                ax.set_ylabel(ylabel, fontsize=11, fontweight='bold', color='#333333')
                ax.tick_params(colors='#666666', labelsize=10)
                ax.legend(loc='best', frameon=True, fancybox=False,
                          edgecolor='#CCCCCC', framealpha=0.95, fontsize=10)

            plt.tight_layout(rect=[0, 0.02, 1, 0.94])

            filename = f"percentage_improvement_{scheme_name}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, facecolor='#FAFAFA', bbox_inches='tight')
            plt.close(fig)

            print(f"Percentage-improvement plot saved to: {filepath}")

    def plot_hop_distribution(self, n_hops_data):
        """
        For each scheme, overlay a histogram of hop counts for completed
        requests, no-shortcut vs shortcut. Median lines marked.

        Parameters
        ----------
        n_hops_data : dict
            n_hops_data[scheme][shortcut_flag][trial] = {uid: hops}
        """

        colors = {0: '#1F77B4', 1: '#FF7F0E'}
        labels = {0: 'No Shortcut', 1: 'Shortcut'}

        for scheme_name, scheme_data in n_hops_data.items():
            hops_pooled = {0: [], 1: []}
            for z in (0, 1):
                for trial_map in scheme_data.get(z, {}).values():
                    hops_pooled[z].extend(trial_map.values())

            if not hops_pooled[0] and not hops_pooled[1]:
                continue

            all_hops = hops_pooled[0] + hops_pooled[1]
            max_hop = max(all_hops) if all_hops else 1
            bins = np.arange(0.5, max_hop + 1.5, 1)

            fig, ax = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor('#FAFAFA')

            # Side-by-side bars: pass both datasets to one ax.hist call so
            # matplotlib places them next to each other within each bin.
            hist_inputs = []
            hist_colors = []
            hist_labels = []
            for z in (0, 1):
                vals = hops_pooled[z]
                if not vals:
                    continue
                hist_inputs.append(vals)
                hist_colors.append(colors[z])
                hist_labels.append(f'{labels[z]} (n={len(vals)}, median={np.median(vals):.0f})')

            ax.hist(hist_inputs, bins=bins, color=hist_colors, label=hist_labels,
                    alpha=0.85, edgecolor='white', linewidth=0.7)

            for z in (0, 1):
                if hops_pooled[z]:
                    ax.axvline(np.median(hops_pooled[z]), color=colors[z],
                               linestyle='--', linewidth=1.5, alpha=0.9, zorder=10)

            ax.set_facecolor('#FFFFFF')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#CCCCCC')
            ax.spines['bottom'].set_color('#CCCCCC')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5,
                    color='#CCCCCC', axis='y')
            ax.set_axisbelow(True)
            ax.set_xticks(range(1, int(max_hop) + 1))

            ax.set_xlabel('Hop Count (per completed request)', fontsize=12,
                          fontweight='bold', color='#333333')
            ax.set_ylabel('Number of Requests (all trials pooled)', fontsize=12,
                          fontweight='bold', color='#333333')
            ax.set_title(
                f'Hop-Count Distribution – {scheme_name}\n'
                f'Dashed lines: median per condition',
                fontsize=13, fontweight='bold', pad=15
            )
            ax.tick_params(colors='#666666', labelsize=10)
            ax.legend(loc='upper right', frameon=True, fancybox=False,
                      edgecolor='#CCCCCC', framealpha=0.95, fontsize=11)

            plt.tight_layout()

            filename = f"hop_distribution_{scheme_name}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, facecolor='#FAFAFA', bbox_inches='tight')
            plt.close(fig)

            print(f"Hop-distribution plot saved to: {filepath}")

    def plot_serve_time_cdf(self, data):
        """
        For each scheme, overlay empirical CDFs of serve_time, no-shortcut
        vs shortcut. CDFs make tails visible in a way the band plot can't.

        Parameters
        ----------
        data : dict
            data[scheme][shortcut_flag][trial] = [(uid, completion_time, serve_time), ...]
        """

        colors = {0: '#1F77B4', 1: '#FF7F0E'}
        labels = {0: 'No Shortcut', 1: 'Shortcut'}

        for scheme_name, scheme_data in data.items():
            serve_pooled = {0: [], 1: []}
            for z in (0, 1):
                for trial_data in scheme_data.get(z, {}).values():
                    for _, _, stime in trial_data:
                        serve_pooled[z].append(stime)

            if not serve_pooled[0] and not serve_pooled[1]:
                continue

            fig, ax = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor('#FAFAFA')

            for z in (0, 1):
                vals = np.array(serve_pooled[z])
                if vals.size == 0:
                    continue
                sorted_vals = np.sort(vals)
                cdf = np.arange(1, sorted_vals.size + 1) / sorted_vals.size
                color = colors[z]
                p50 = np.percentile(sorted_vals, 50)
                p95 = np.percentile(sorted_vals, 95)
                p99 = np.percentile(sorted_vals, 99)
                ax.plot(sorted_vals, cdf, color=color, linewidth=2.2,
                        label=(f'{labels[z]}  median={p50:.1f}, '
                               f'p95={p95:.1f}, p99={p99:.1f}'))

            for q, style in [(0.5, ':'), (0.95, ':'), (0.99, ':')]:
                ax.axhline(q, color='#666666', linestyle=style,
                           linewidth=0.8, alpha=0.5, zorder=1)

            ax.set_facecolor('#FFFFFF')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#CCCCCC')
            ax.spines['bottom'].set_color('#CCCCCC')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='#CCCCCC')
            ax.set_axisbelow(True)
            ax.set_ylim(0, 1.005)

            ax.set_xlabel('Service Time', fontsize=12, fontweight='bold', color='#333333')
            ax.set_ylabel('Empirical CDF', fontsize=12, fontweight='bold', color='#333333')
            ax.set_title(
                f'Service-Time CDF – {scheme_name}\n'
                f'Curve further right at a given y means slower at that percentile',
                fontsize=13, fontweight='bold', pad=15
            )
            ax.tick_params(colors='#666666', labelsize=10)
            ax.legend(loc='lower right', frameon=True, fancybox=False,
                      edgecolor='#CCCCCC', framealpha=0.95, fontsize=10)

            plt.tight_layout()

            filename = f"serve_time_cdf_{scheme_name}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, facecolor='#FAFAFA', bbox_inches='tight')
            plt.close(fig)

            print(f"Serve-time CDF plot saved to: {filepath}")

    def plot_speedup_vs_hops(self, data, n_hops_data, min_samples=5):
        """
        For each scheme, group requests by their *no-shortcut* hop count and
        box-plot the per-request % speedup. Two variants emitted per scheme:

          - completion-time speedup: dominated by cumulative throughput,
            tends to look flat across hop counts.
          - serve-time speedup: per-request routing cost. Reveals whether
            "harder" (more-hop) requests benefit more.

        Pairing is uid- and trial-exact.

        Parameters
        ----------
        data : dict
            data[scheme][shortcut_flag][trial] = [(uid, completion_time, serve_time), ...]
        n_hops_data : dict
            n_hops_data[scheme][shortcut_flag][trial] = {uid: hops}
        min_samples : int
            Drop hop-count bins with fewer than this many paired observations.
        """

        # (metric_key, tuple_index, filename_suffix, ylabel, title_suffix)
        metrics = [
            ('completion', 1, 'completion',
             'Completion-Time Speedup (%) — (no_sc − sc) / no_sc',
             'Cumulative completion time (throughput-dominated)'),
            ('serve', 2, 'serve',
             'Serve-Time Speedup (%) — (no_sc − sc) / no_sc',
             'Per-request serve time (routing-cost view)'),
        ]

        for scheme_name, scheme_data in data.items():
            n_hops_scheme = n_hops_data.get(scheme_name, {})

            no_trials = scheme_data.get(0, {})
            sc_trials = scheme_data.get(1, {})
            shared_trials = sorted(set(no_trials.keys()) & set(sc_trials.keys()))

            for _, tup_idx, suffix, ylabel, title_suffix in metrics:
                speedups_by_hops = {}

                for trial_idx in shared_trials:
                    no_data = no_trials.get(trial_idx, [])
                    sc_data = sc_trials.get(trial_idx, [])
                    no_hops_map = n_hops_scheme.get(0, {}).get(trial_idx, {})
                    if not no_data or not sc_data:
                        continue

                    no_by_uid = {tup[0]: tup[tup_idx] for tup in no_data}
                    sc_by_uid = {tup[0]: tup[tup_idx] for tup in sc_data}
                    common = set(no_by_uid) & set(sc_by_uid) & set(no_hops_map)
                    for uid in common:
                        no_v = no_by_uid[uid]
                        sc_v = sc_by_uid[uid]
                        hops = no_hops_map[uid]
                        if no_v <= 0:
                            continue
                        pct = (no_v - sc_v) / no_v * 100
                        speedups_by_hops.setdefault(hops, []).append(pct)

                bins = sorted(h for h, vals in speedups_by_hops.items() if len(vals) >= min_samples)
                if not bins:
                    continue

                box_data = [speedups_by_hops[h] for h in bins]
                counts = [len(speedups_by_hops[h]) for h in bins]
                medians = [float(np.median(speedups_by_hops[h])) for h in bins]

                fig, ax = plt.subplots(figsize=(14, 6))
                fig.patch.set_facecolor('#FAFAFA')

                bp = ax.boxplot(
                    box_data, positions=bins, widths=0.6,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(linewidth=1.4, edgecolor='#666666'),
                    whiskerprops=dict(linewidth=1.2, color='#666666'),
                    capprops=dict(linewidth=1.2, color='#666666'),
                    medianprops=dict(linewidth=2.2, color='#D62728'),
                )
                for patch in bp['boxes']:
                    patch.set_facecolor('#FF7F0E')
                    patch.set_alpha(0.55)

                ax.plot(bins, medians, color='#D62728', linewidth=1.8,
                        marker='o', markersize=6, alpha=0.85,
                        label='Median speedup per bin')

                ymax = max(np.percentile(vals, 95) for vals in box_data)
                for h, n, m in zip(bins, counts, medians):
                    ax.text(h, ymax * 1.05, f'n={n}', ha='center', va='bottom',
                            fontsize=9, color='#333333')

                ax.axhline(y=0, color='#333333', linestyle='--', linewidth=1.2,
                           alpha=0.7, zorder=5)

                ax.set_facecolor('#FFFFFF')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#CCCCCC')
                ax.spines['bottom'].set_color('#CCCCCC')
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5,
                        color='#CCCCCC', axis='y')
                ax.set_axisbelow(True)
                ax.set_xticks(bins)

                ax.set_xlabel('No-Shortcut Hop Count (request "difficulty")',
                              fontsize=12, fontweight='bold', color='#333333')
                ax.set_ylabel(ylabel, fontsize=12, fontweight='bold', color='#333333')
                ax.set_title(
                    f'Shortcut Speedup vs Request Difficulty – {scheme_name}\n'
                    f'{title_suffix}. Paired by uid+trial; '
                    f'bins with <{min_samples} samples dropped',
                    fontsize=12, fontweight='bold', pad=15
                )
                ax.tick_params(colors='#666666', labelsize=10)
                ax.legend(loc='best', frameon=True, fancybox=False,
                          edgecolor='#CCCCCC', framealpha=0.95, fontsize=10)

                plt.tight_layout()

                filename = f"speedup_vs_hops_{suffix}_{scheme_name}.png"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300, facecolor='#FAFAFA', bbox_inches='tight')
                plt.close(fig)

                print(f"Speedup-vs-hops ({suffix}) plot saved to: {filepath}")

    def plot_od_speedup_heatmap(self, data, pair_lookup, net_size=None,
                                 min_samples=10, colorbar_clip=50.0):
        """
        Per-pair (origin → destination) heatmap of mean serve-time speedup.
        Cells with fewer than ``min_samples`` paired observations are shown
        gray. The colorbar is clipped to ±``colorbar_clip`` percent so a few
        extreme outliers don't wash out the moderate-but-systematic pattern.

        Diverging colormap centered at 0 (red = shortcut hurts, blue = helps).

        Parameters
        ----------
        data : dict
            data[scheme][shortcut_flag][trial] = [(uid, completion_time, serve_time), ...]
        pair_lookup : dict
            pair_lookup[trial_idx][uid] = (origin, destination)
        net_size : int or None
            Side length of the heatmap. If None, inferred from pair_lookup.
        min_samples : int
            Cells with fewer paired observations are dropped (shown gray).
        colorbar_clip : float
            Symmetric colorbar limit in percent. Cells beyond this still
            display at the colorbar extreme; the underlying value isn't
            changed.
        """

        if net_size is None:
            all_nodes = set()
            for trial_map in pair_lookup.values():
                for o, d in trial_map.values():
                    all_nodes.add(o)
                    all_nodes.add(d)
            net_size = max(all_nodes) + 1 if all_nodes else 0
        if net_size == 0:
            return

        for scheme_name, scheme_data in data.items():
            no_trials = scheme_data.get(0, {})
            sc_trials = scheme_data.get(1, {})
            shared = sorted(set(no_trials.keys()) & set(sc_trials.keys()))

            accum = {}
            for trial_idx in shared:
                no_data = no_trials.get(trial_idx, [])
                sc_data = sc_trials.get(trial_idx, [])
                pair_map = pair_lookup.get(trial_idx, {})
                if not no_data or not sc_data or not pair_map:
                    continue
                no_st = {uid: st for uid, _, st in no_data}
                sc_st = {uid: st for uid, _, st in sc_data}
                common = set(no_st) & set(sc_st) & set(pair_map)
                for uid in common:
                    no_v = no_st[uid]
                    sc_v = sc_st[uid]
                    if no_v <= 0:
                        continue
                    pct = (no_v - sc_v) / no_v * 100
                    o, d = pair_map[uid]
                    accum.setdefault((o, d), []).append(pct)

            if not accum:
                continue

            mat = np.full((net_size, net_size), np.nan)
            counts = np.zeros((net_size, net_size), dtype=int)
            for (o, d), vals in accum.items():
                if len(vals) < min_samples:
                    continue
                mat[o, d] = float(np.mean(vals))
                counts[o, d] = len(vals)

            finite = mat[~np.isnan(mat)]
            if finite.size == 0:
                continue
            vmax = float(colorbar_clip)

            fig, ax = plt.subplots(figsize=(11, 9))
            fig.patch.set_facecolor('#FAFAFA')

            # Gray background for cells without data
            ax.set_facecolor('#EEEEEE')
            masked = np.ma.masked_invalid(mat)
            cmap = plt.get_cmap('RdBu_r').copy()
            cmap.set_bad('#EEEEEE')

            im = ax.imshow(masked, cmap=cmap, vmin=-vmax, vmax=vmax,
                           origin='upper', aspect='equal')

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Mean Serve-Time Speedup (%)',
                           fontsize=11, fontweight='bold')

            ax.set_xticks(range(net_size))
            ax.set_yticks(range(net_size))
            ax.set_xticklabels(range(net_size), fontsize=8)
            ax.set_yticklabels(range(net_size), fontsize=8)
            ax.set_xlabel('Destination node', fontsize=12, fontweight='bold',
                          color='#333333')
            ax.set_ylabel('Origin node', fontsize=12, fontweight='bold',
                          color='#333333')
            ax.set_title(
                f'Per-Pair Serve-Time Speedup – {scheme_name}\n'
                f'Mean (no_sc − sc) / no_sc × 100% per O→D pair. '
                f'Colorbar clipped to ±{vmax:.0f}%; cells with <{min_samples} obs gray.',
                fontsize=12, fontweight='bold', pad=12
            )

            plt.tight_layout()

            filename = f"od_speedup_heatmap_{scheme_name}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, facecolor='#FAFAFA', bbox_inches='tight')
            plt.close(fig)

            print(f"O→D speedup heatmap saved to: {filepath}")

    def plot_speedup_by_cluster(self, data, pair_lookup, graph_arr, min_samples=10):
        """
        Auto-detect communities on the graph (greedy modularity), then plot
        mean serve-time speedup per (source_community → destination_community)
        pair as a bar chart with 95% CI error bars.

        Far fewer bars than the 20×20 heatmap, so the cross-cluster signal
        isn't drowned in per-pair noise. For a barbell, expect two large
        intra-cluster bars near 0% and the cross-cluster bars carrying the
        positive speedup.

        Parameters
        ----------
        data : dict
            data[scheme][shortcut_flag][trial] = [(uid, completion_time, serve_time), ...]
        pair_lookup : dict
            pair_lookup[trial_idx][uid] = (origin, destination)
        graph_arr : np.ndarray
            Adjacency matrix used to detect communities.
        min_samples : int
            Drop (src_comm, dst_comm) groups with fewer paired observations.
        """

        import networkx as nx

        G = nx.Graph(graph_arr)
        try:
            communities = list(nx.community.greedy_modularity_communities(G))
        except Exception:
            communities = [set(G.nodes)]
        # Sort communities by lowest node so labels are stable across runs.
        communities = sorted([sorted(c) for c in communities], key=lambda c: c[0])

        node_to_comm = {}
        for ci, comm in enumerate(communities):
            for n in comm:
                node_to_comm[n] = ci
        num_comms = len(communities)

        # Human-readable label for each community: "C<idx> {n0,n1,...}".
        comm_labels = [
            f"C{ci} {{{','.join(str(n) for n in comm)}}}"
            for ci, comm in enumerate(communities)
        ]

        for scheme_name, scheme_data in data.items():
            no_trials = scheme_data.get(0, {})
            sc_trials = scheme_data.get(1, {})
            shared = sorted(set(no_trials.keys()) & set(sc_trials.keys()))

            accum = {(i, j): [] for i in range(num_comms) for j in range(num_comms)}
            for trial_idx in shared:
                no_data = no_trials.get(trial_idx, [])
                sc_data = sc_trials.get(trial_idx, [])
                pair_map = pair_lookup.get(trial_idx, {})
                if not no_data or not sc_data or not pair_map:
                    continue
                no_st = {uid: st for uid, _, st in no_data}
                sc_st = {uid: st for uid, _, st in sc_data}
                common = set(no_st) & set(sc_st) & set(pair_map)
                for uid in common:
                    no_v = no_st[uid]
                    sc_v = sc_st[uid]
                    if no_v <= 0:
                        continue
                    o, d = pair_map[uid]
                    if o not in node_to_comm or d not in node_to_comm:
                        continue
                    pct = (no_v - sc_v) / no_v * 100
                    accum[(node_to_comm[o], node_to_comm[d])].append(pct)

            groups = [(k, v) for k, v in accum.items() if len(v) >= min_samples]
            if not groups:
                continue
            groups.sort(key=lambda kv: kv[0])

            x_labels, means, ci_half, counts, colors_bar = [], [], [], [], []
            for (i, j), vals in groups:
                arr = np.array(vals)
                mean = float(arr.mean())
                sem = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
                means.append(mean)
                ci_half.append(1.96 * sem)
                counts.append(arr.size)
                x_labels.append(f"C{i}→C{j}" + (" (intra)" if i == j else ""))
                colors_bar.append('#1F77B4' if i == j else '#FF7F0E')

            x_pos = np.arange(len(groups))
            fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(groups)), 6))
            fig.patch.set_facecolor('#FAFAFA')

            bars = ax.bar(x_pos, means, yerr=ci_half, capsize=4,
                          color=colors_bar, alpha=0.85,
                          edgecolor='white', linewidth=0.8,
                          error_kw=dict(ecolor='#333333', linewidth=1.2))
            ax.axhline(0, color='#333333', linestyle='--', linewidth=1.2,
                       alpha=0.7, zorder=5)

            for bar, mean, n in zip(bars, means, counts):
                ax.text(bar.get_x() + bar.get_width() / 2.,
                        bar.get_height() + (1 if mean >= 0 else -1) * 0.5,
                        f'{mean:+.1f}%\n(n={n})',
                        ha='center', va='bottom' if mean >= 0 else 'top',
                        fontsize=9, color='#333333')

            ax.set_facecolor('#FFFFFF')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#CCCCCC')
            ax.spines['bottom'].set_color('#CCCCCC')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5,
                    color='#CCCCCC', axis='y')
            ax.set_axisbelow(True)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=10)
            ax.set_ylabel('Mean Serve-Time Speedup (%) — (no_sc − sc) / no_sc',
                          fontsize=11, fontweight='bold', color='#333333')

            legend_lines = "  |  ".join(comm_labels)
            ax.set_title(
                f'Speedup by Community Pair – {scheme_name}\n'
                f'Auto-detected communities (greedy modularity).  {legend_lines}\n'
                f'Bars: mean ± 95% CI; blue = intra-community, orange = cross-community.',
                fontsize=11, fontweight='bold', pad=15
            )
            ax.tick_params(colors='#666666', labelsize=10)

            plt.tight_layout()
            filename = f"speedup_by_cluster_{scheme_name}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, facecolor='#FAFAFA', bbox_inches='tight')
            plt.close(fig)

            print(f"Speedup-by-cluster plot saved to: {filepath}")

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

        filename = f"shortcut_usage.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, facecolor='#FAFAFA', bbox_inches='tight')
        plt.close(fig)

        print(f"Shortcut usage plot saved to: {filepath}")

        # Create a second plot: comparison across schemes
        self._plot_shortcut_comparison(shortcut_data, total_requests_data)

    def _plot_shortcut_comparison(self, shortcut_data, total_requests_data):
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

        filename = f"shortcut_comparison.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, facecolor='#FAFAFA', bbox_inches='tight')
        plt.close(fig)

        print(f"Shortcut comparison plot saved to: {filepath}")