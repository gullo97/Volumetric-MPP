import matplotlib.pyplot as plt


def plot_volumetric_persistence_barcode(peaks_info, xlabel="Channel", ylabel="Persistence", title="Barcode Persistence Plot", COLORS=None):
    """
    Generate a barcode plot for peak volumetric persistence.
    
    Each detected peak is represented by a vertical line whose height corresponds to its persistence.
    """
    FONTSIZE = {
        "label":  14,
        "title":  16,
        "legend": 12,
    }
    if COLORS is None:
        # first k colors from the default color cycle, then default color
        COLORS = {
            "default": "#8499C4",
            "detected_peak": ["#C94040", "#69995D", "#CBAC88", "#394648"],
            "alphas":    [1.0, 1.0, 1.0, 1.0],
            "linewidths": [2.0, 2.0, 2.0, 2.0],
        }
    

    plt.figure(figsize=(7, 2.5))
    for rank, cand in enumerate(peaks_info):
        if rank < len(COLORS['detected_peak']):
            plt.vlines(cand['peak_index'], 0, cand['persistence'], colors=COLORS['detected_peak'][rank],
                       lw=COLORS['linewidths'][rank], alpha=COLORS['alphas'][rank])
            plt.scatter(cand['peak_index'], cand['persistence'], color=COLORS['detected_peak'][rank],
                         s=60, edgecolor='k', zorder=3)
        else:
            plt.vlines(cand['peak_index'], 0, cand['persistence'], colors=COLORS['default'],
                       lw=COLORS['linewidths'][0], alpha=COLORS['alphas'][0])
            plt.scatter(cand['peak_index'], cand['persistence'], color=COLORS['default'],
                         s=60, edgecolor='k', zorder=3)
    # dynamically set x-axis limit based on max detected peak index
    if peaks_info:
        max_idx = max(cand['peak_index'] for cand in peaks_info) + 50
        plt.xlim(0, max_idx)
    # plt.ylim(0, max(cand['persistence'] for cand in peaks_info) * 1.1)
    plt.xlabel(xlabel, fontsize=FONTSIZE["label"])
    plt.ylabel(ylabel, fontsize=FONTSIZE["label"])
    # plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()



def plot_spectrum_with_detected_peaks(
    x,
    spectrum,
    peaks_info,
    top_k=3,
    detected_style='cross',            # or 'vertical_line'
    COLORS=None,
    FONTSIZE=None,
):
    """
    Aesthetic spectrum + detected‐peaks plot with customizable colors & font‐sizes.
    
    Now the legend omits the spectrum line and instead shows each detected peak
    as "Rank # Peak" with its corresponding color.
    """
    import matplotlib.pyplot as plt
    
    # Defaults
    if COLORS is None:
        COLORS = {
            "spectrum":      plt.rcParams['axes.prop_cycle'].by_key()['color'][0],
            "detected_peak": ['red', 'orange', 'green', 'purple'],
            "alphas":    [1.0, 1.0, 1.0, 1.0],
            "linewidths": [2.0, 2.0, 2.0, 1.5],
        }
    if FONTSIZE is None:
        FONTSIZE = {
            "label":  plt.rcParams['axes.labelsize'],
            "title":  plt.rcParams['axes.titlesize'],
            "legend": plt.rcParams['legend.fontsize'],
        }

    # select top_k peaks (assumed already sorted by persistence)
    selected = peaks_info[:top_k]
    detected_idxs = [int(c['peak_index']) for c in selected]

    fig, ax = plt.subplots(figsize=(7, 2.5))
    # plot spectrum without legend entry
    ax.plot(x, spectrum, lw=1, color=COLORS["spectrum"], label='_nolegend_')

    if detected_style == "cross":
        for rank, idx in enumerate(detected_idxs, start=1):
            y = spectrum[idx]
            ax.scatter(
                idx, y,
                marker='x', s=100,
                color=COLORS["detected_peak"][(rank-1) % len(COLORS["detected_peak"])],
                label=f"Rank {rank} Peak",
                zorder=3
            )

    elif detected_style == "vertical_line":
        for rank, idx in enumerate(detected_idxs, start=1):
            ax.axvline(
                idx,
                color=COLORS["detected_peak"][(rank-1) % len(COLORS["detected_peak"])],
                linestyle="--", linewidth=COLORS["linewidths"][(rank-1) % len(COLORS["linewidths"])],
                alpha=COLORS["alphas"][(rank-1) % len(COLORS["alphas"])],
                label=f"Rank {rank} Peak"
            )
    else:
        raise ValueError("detected_style must be 'cross' or 'vertical_line'")

    # Styling
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel("Channel", fontsize=FONTSIZE["label"])
    ax.set_ylabel("Intensity", fontsize=FONTSIZE["label"])
    # ax.legend(
    #     ncol=1,
    #     fontsize=FONTSIZE["legend"],
    #     frameon=True,
    #     framealpha=1.0
    # )
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt


def plot_candidate_inner_grid(candidate, outer_smoothing, outer_bins, fixed_params, 
                              view_type='points', colormap='viridis',
                              use_frequency_colormap=True, single_color='C0',
                              full_ranges=None):
    """
    Unified visualization for the candidate's inner grid search volume.
    
    Parameters:
        candidate (dict): Candidate dictionary from find_peaks_volumetric_persistence.
        outer_smoothing (int): Smoothing window size to filter candidate's data.
        outer_bins (int): Bins_factor to filter candidate's data.
        fixed_params (dict): Dictionary of inner grid parameter(s) to fix 
                             (e.g. {'distance': 10}). Must fix at least one parameter.
        view_type (str): 'points' for a scatter (point cloud) view or 'voxels' for a voxel rendering.
        colormap (str): Colormap to use when `use_frequency_colormap=True`.
        use_frequency_colormap (bool): Whether to color points/voxels by frequency (True)
                                       or use a single color (False).
        single_color (str): Matplotlib color code to use when `use_frequency_colormap=False`.
        full_ranges (dict, optional): Dict mapping free parameter names to (min, max) limits.
        
    The inner grid originally spans the parameters:
        ['threshold', 'width', 'prominence', 'distance']
    After filtering by outer parameters and by fixed_params, the remaining ("free")
    parameters form a lower-dimensional space (1D, 2D, or 3D) that will be plotted.
    """
    

    # Define the full set of inner grid keys.
    inner_keys = ['threshold', 'width', 'prominence', 'distance']

    # Filter by outer parameters
    filtered = [params for params in candidate['grid_params']
                if params['smoothing'] == outer_smoothing and params['bins_factor'] == outer_bins]
    # Filter by fixed inner parameters
    for key, val in fixed_params.items():
        filtered = [params for params in filtered if np.isclose(params[key], val)]
    if not filtered:
        print("No inner grid data for the specified outer parameters and fixed settings.")
        return

    # Determine free parameters
    free_params = [key for key in inner_keys if key not in fixed_params]
    dim = len(free_params)
    if dim == 0:
        print(f"All inner grid parameters fixed. Frequency: {len(filtered)}")
        return
    if dim > 3:
        print("Too many free parameters to plot. Please fix additional parameters.")
        return

    # Count frequency
    from collections import Counter
    combo_counts = Counter()
    for params in filtered:
        key = tuple(params[p] for p in free_params)
        combo_counts[key] += 1

    coords = list(combo_counts.keys())
    counts = list(combo_counts.values())

    if view_type == 'points':
        # Prepare common label and colormap behavior
        def plot_scatter(ax, xs, ys, zs=None):
            if use_frequency_colormap:
                cargs = dict(c=counts, cmap=colormap)
                show_cb = True
            else:
                cargs = dict(color=single_color)
                show_cb = False
            sizes = [c * 50 for c in counts]
            if dim == 3:
                sc = ax.scatter(xs, ys, zs, s=sizes, alpha=0.8, **cargs)
            else:
                sc = ax.scatter(xs, ys, s=sizes, alpha=0.8, **cargs)
            if show_cb:
                plt.colorbar(sc, ax=ax, label="Frequency")
            return sc

        if dim == 3:
            x_vals, y_vals, z_vals = zip(*coords)
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            plot_scatter(ax, x_vals, y_vals, z_vals)
            ax.set_xlabel(free_params[0])
            ax.set_ylabel(free_params[1])
            ax.set_zlabel(free_params[2])
            if full_ranges:
                if free_params[0] in full_ranges: ax.set_xlim(full_ranges[free_params[0]])
                if free_params[1] in full_ranges: ax.set_ylim(full_ranges[free_params[1]])
                if free_params[2] in full_ranges: ax.set_zlim(full_ranges[free_params[2]])
        elif dim == 2:
            x_vals, y_vals = zip(*coords)
            fig, ax = plt.subplots(figsize=(10, 7))
            plot_scatter(ax, x_vals, y_vals)
            ax.set_xlabel(free_params[0])
            ax.set_ylabel(free_params[1])
            if full_ranges:
                if free_params[0] in full_ranges: ax.set_xlim(full_ranges[free_params[0]])
                if free_params[1] in full_ranges: ax.set_ylim(full_ranges[free_params[1]])
        else:  # dim == 1
            x_vals = [c[0] for c in coords]
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.stem(x_vals, counts, use_line_collection=True, linefmt=single_color, markerfmt=single_color)
            ax.set_xlabel(free_params[0])
            ax.set_ylabel("Frequency")
            if full_ranges and free_params[0] in full_ranges:
                ax.set_xlim(full_ranges[free_params[0]])
        plt.title(f"Inner Grid (Points) for Peak\nSmoothing={outer_smoothing}, bins={outer_bins}, fixed={fixed_params}")
        plt.tight_layout()
        plt.show()

    elif view_type == 'voxels':
        if dim != 3:
            print("Voxel view requires exactly 3 free parameters.")
            return
        # Build grid
        unique_vals = {param: sorted({coord[i] for coord in coords})
                       for i, param in enumerate(free_params)}
        shape = tuple(len(unique_vals[p]) for p in free_params)
        freq_array = np.zeros(shape)
        idx_map = {p: {v: i for i, v in enumerate(unique_vals[p])} for p in free_params}
        for key, cnt in combo_counts.items():
            i, j, k = (idx_map[free_params[d]][key[d]] for d in range(3))
            freq_array[i, j, k] = cnt
        occupancy = freq_array > 0

        # Compute colors
        if use_frequency_colormap:
            import matplotlib.colors as mcolors
            norm = mcolors.Normalize(vmin=freq_array[occupancy].min(), vmax=freq_array[occupancy].max())
            cmap = plt.get_cmap(colormap)
            facecolors = np.empty(shape, dtype=object)
            for idx, val in np.ndenumerate(freq_array):
                if occupancy[idx]:
                    facecolors[idx] = cmap(norm(val))
                else:
                    facecolors[idx] = (0,0,0,0)
        else:
            facecolors = np.full(shape, single_color, dtype=object)

        # Meshgrid
        def edges(arr):
            d = (arr[1] - arr[0]) if len(arr)>1 else 1
            return np.concatenate([arr - d/2, [arr[-1] + d/2]])
        edges_dict = {p: edges(unique_vals[p]) for p in free_params}
        X, Y, Z = np.meshgrid(edges_dict[free_params[0]],
                              edges_dict[free_params[1]],
                              edges_dict[free_params[2]], indexing='ij')

        fig = plt.figure(figsize=(4.2, 3.5))
        ax = fig.add_subplot(111, projection='3d')
        ax.voxels(X, Y, Z, occupancy, facecolors=facecolors, edgecolor='k', linewidth=0.5)
        ax.set_xlabel(free_params[0])
        ax.set_ylabel(free_params[1])
        ax.set_zlabel(free_params[2])
        if full_ranges:
            ax.set_xlim(full_ranges.get(free_params[0], ax.get_xlim()))
            ax.set_ylim(full_ranges.get(free_params[1], ax.get_ylim()))
            ax.set_zlim(full_ranges.get(free_params[2], ax.get_zlim()))

        if use_frequency_colormap:
            from matplotlib.cm import ScalarMappable
            mappable = ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array(freq_array[occupancy])
            plt.colorbar(mappable, ax=ax, shrink=0.5, pad=0.1, label="Frequency")

        # plt.title(f"Inner Grid Voxels for Peak\nSmoothing={outer_smoothing}, bins={outer_bins}, fixed={fixed_params}")


        # set font sizes
        # ax.xaxis.label.set_size(FONTSIZE["label"])
        # ax.yaxis.label.set_size(FONTSIZE["label"])
        # ax.zaxis.label.set_size(FONTSIZE["label"])
        # ax.title.set_size(FONTSIZE["title"])
        # ax.tick_params(axis='both', which='major', labelsize=FONTSIZE["label"])

        # # remove axis labels
        # ax.set_xlabel('')
        # ax.set_ylabel('')
        # ax.set_zlabel('')

        # set legend font size
        plt.tight_layout()
        plt.show()


def plot_multi_volumetric_persistence_radar_polygon(
    candidates,
    param_names=None,
    explored_ranges=None,
    colors=None,
    labels=None,
    title="Volumetric Persistence Radar",
    font_size=12,
    line_width=2,
    grid_line_style="--",
    grid_line_width=0.8,
    grid_color="grey",
    fill_alpha=0.1,
    label_pad=15
):
    """
    Publication-ready radar chart overlaying volumetric persistence for multiple peaks,
    with polygonal grid matching the number of axes, Times New Roman font, and
    customizable grid color, label padding, and no outer frame circle.

    Only the polygonal grid lines are shown (no circular grid).
    """
    # Use Times New Roman for all text
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # Default parameter list
    data_params = param_names or ['smoothing', 'bins_factor', 'threshold', 'width', 'prominence', 'distance']
    N = len(data_params)
    
    # Compute angles for axes
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    # Display labels: Title case
    display_labels = [p.replace('_', ' ').title() for p in data_params]
    
    # Colors and legend labels
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if labels is None:
        labels = [str(c.get('peak_index', idx)) for idx, c in enumerate(candidates)]
    
    # Create polar plot
    fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw=dict(polar=True))
    
    # Disable default circular grid and frame
    ax.grid(False)
    ax.spines['polar'].set_visible(False)
    
    # Draw polygonal grid for radial ticks only
    radial_ticks = [0.25, 0.5, 0.75, 1.0]
    for r in radial_ticks:
        ax.plot(angles, [r]*len(angles),
                linestyle=grid_line_style,
                linewidth=grid_line_width,
                color=grid_color,
                zorder=0)
    
    # Draw radial spokes
    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, 1],
                color=grid_color, linewidth=grid_line_width, zorder=0)
    
    # Plot each candidate
    for idx, cand in enumerate(candidates):
        # Compute normalized values per axis
        vals = []
        for param in data_params:
            vlist = sorted({p[param] for p in cand['grid_params']})
            if len(vlist) < 2:
                vals.append(0.0)
                continue
            d = np.diff(vlist)
            thr = d.min() * 1.5
            segs = []
            start = vlist[0]
            for i, diff in enumerate(d):
                if diff > thr:
                    segs.append((start, vlist[i]))
                    start = vlist[i+1]
            segs.append((start, vlist[-1]))
            length = sum(e - s for s, e in segs)
            lo, hi = explored_ranges.get(param, (min(vlist), max(vlist))) if explored_ranges else (min(vlist), max(vlist))
            rng = hi - lo
            vals.append(length / rng if rng > 0 else 0.0)
        vals += vals[:1]
        
        color = colors[idx % len(colors)]
        ax.plot(angles, vals, color=color, linewidth=line_width, label=labels[idx], zorder=len(candidates) - idx)
        # if idx < len(candidates) - 1:
        ax.fill(angles, vals, color=color, alpha=fill_alpha)
    
    # Set axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(display_labels, fontsize=font_size)
    ax.tick_params(axis='x', pad=label_pad)
    
    # Radial ticks labels
    # ax.set_yticks(radial_ticks)
    # ax.set_yticklabels([f"{int(r*100)}%" for r in radial_ticks], fontsize=int(font_size*0.9))
    # nullify radial ticks
    ax.set_yticklabels([])
    ax.set_yticks([])
    
    # Title and legend
    ax.set_title(title, fontsize=int(font_size*1.2), pad=20)
    # legend = ax.legend()
    # legend.get_frame().set_alpha(0.5)
    
    # Set radial limit
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
