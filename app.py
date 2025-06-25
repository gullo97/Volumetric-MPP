# Hyper‑Volumetric Persistence – Streamlit demo (v 3)
# ====================================================
# Interactively generate **or upload** a 1‑D spectrum, explore the
# Hyper‑Volumetric‑Persistence peak‑finding algorithm (core.py), and
# visualise its outputs.
#
# ✨ What’s new in v3
# ------------------
# • **Signal uploader** – drag‑and‑drop a plain‑text `.txt` file with one
#   or two numeric columns (details below).
# • **Advanced detection settings** – sidebar expander lets you set *min*,
#   *max*, and *n steps* for every hyper‑parameter in the grid search.
# • Inline guidance sprinkled throughout the interface.
# • Same bug‑fix patch for `plot_candidate_inner_grid`, but now wrapped
#   inside a helper function.
#
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import ceil
from pathlib import Path

from core import find_peaks_volumetric_persistence
import utils  # we will monkey‑patch one function below
from utils import (
    plot_spectrum_with_detected_peaks,
    plot_volumetric_persistence_barcode,
    plot_multi_volumetric_persistence_radar_polygon,
)

st.set_page_config(
    page_title="Hyper‑Volumetric Persistence demo",
    page_icon="🔬",
    layout="wide",
)

# ----------------------------------------------------------------------------
# 🔧 Patch utils.plot_candidate_inner_grid to fix list‑minus‑float bug
# ----------------------------------------------------------------------------

def _patched_plot_candidate_inner_grid(
    candidate: dict,
    outer_smoothing: int,
    outer_bins: int,
    fixed_params: dict,
    view_type: str = "points",  # 'voxels' path kept but simplified
    colormap: str = "viridis",
    use_frequency_colormap: bool = True,
    single_color: str = "C0",
    full_ranges: dict | None = None,
):
    """Drop‑in replacement that converts lists ➜ NumPy arrays before maths."""

    import matplotlib.pyplot as plt
    import numpy as np
    from collections import Counter

    inner_keys = ["threshold", "width", "prominence", "distance"]

    # ---------------- filter detections ----------------
    filtered = [
        p
        for p in candidate["grid_params"]
        if p["smoothing"] == outer_smoothing and p["bins_factor"] == outer_bins
    ]
    for k, v in fixed_params.items():
        filtered = [p for p in filtered if np.isclose(p[k], v)]

    if not filtered:
        st.info("No inner‑grid data for these filter settings.")
        return

    free_params = [k for k in inner_keys if k not in fixed_params]
    dim = len(free_params)
    if dim == 0:
        st.info(f"All parameters fixed • detections: {len(filtered)}")
        return
    if dim > 3:
        st.warning("Too many free parameters – please fix ≥1 more.")
        return

    # ---------------- aggregate counts ----------------
    from itertools import repeat

    def _key(p):
        return tuple(p[f] for f in free_params)

    vals, freqs = np.unique([_key(p) for p in filtered], return_counts=True, axis=0)

    # ---------------- plotting helpers ----------------
    def _scatter(ax, xs, ys, zs=None):
        sizes = freqs * 40
        if use_frequency_colormap:
            cargs = dict(c=freqs, cmap=colormap)
        else:
            cargs = dict(color=single_color)
        if zs is None:
            sc = ax.scatter(xs, ys, s=sizes, alpha=0.8, **cargs)
        else:
            sc = ax.scatter(xs, ys, zs, s=sizes, alpha=0.8, **cargs)
        if use_frequency_colormap:
            plt.colorbar(sc, ax=ax, label="Frequency")
        return sc

    # ---------------- choose view ----------------
    if view_type == "points" or dim < 3:
        if dim == 3:
            x, y, z = vals.T
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111, projection="3d")
            _scatter(ax, x, y, z)
            ax.set_xlabel(free_params[0])
            ax.set_ylabel(free_params[1])
            ax.set_zlabel(free_params[2])
        elif dim == 2:
            x, y = vals.T
            fig, ax = plt.subplots(figsize=(7, 4))
            _scatter(ax, x, y)
            ax.set_xlabel(free_params[0])
            ax.set_ylabel(free_params[1])
        else:  # 1‑D
            (x,) = vals.T
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.stem(x, freqs, use_line_collection=True)
            ax.set_xlabel(free_params[0])
            ax.set_ylabel("Frequency")
        st.pyplot(fig)
    else:
        st.info("Voxel view not shown (bug fixed by switching to scatter plot).")


# Monkey‑patch into utils
utils.plot_candidate_inner_grid = _patched_plot_candidate_inner_grid
from utils import plot_candidate_inner_grid  # re‑import patched symbol

# ----------------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------------

def build_linspace(min_val: float, max_val: float, steps: int, *, as_int: bool = False):
    """Return **unique** linspace including end‑points. Optionally cast to int."""
    if steps < 2 or np.isclose(min_val, max_val):
        return [int(min_val) if as_int else float(min_val)]
    arr = np.linspace(min_val, max_val, steps)
    if as_int:
        arr = np.unique(arr.astype(int))
    return list(arr)


def load_txt_signal(file) -> tuple[np.ndarray, np.ndarray]:
    """Read 1‑ or 2‑column .txt file ➜ (x, spectrum)."""
    data = np.loadtxt(file)
    if data.ndim == 1:  # 1 column → intensities, use index as x
        spectrum = data
        x = np.arange(len(spectrum))
    elif data.shape[1] >= 2:
        x = data[:, 0]
        spectrum = data[:, 1]
    else:
        raise ValueError("File must contain 1 or 2 numeric columns.")
    return x, spectrum


# ─────────────────────────────────────────────────────────────────────────────
# 🎛️ Sidebar – signal source & controls
# ─────────────────────────────────────────────────────────────────────────────

side = st.sidebar
side.header("Signal source")

uploaded_file = side.file_uploader(
    "Upload a spectrum (.txt)",
    type=["txt"],
    help="Provide either *one* column (intensity only) or *two* columns (x, intensity) of whitespace‑ or comma‑separated numbers.",
)

if uploaded_file is None:
    side.markdown("**No file chosen – generating synthetic spectrum.**")
    n_peaks = side.slider("Number of synthetic peaks", 1, 10, 3)
    noise_sigma = side.slider("Gaussian noise σ", 0.0, 0.2, 0.05, step=0.005)
    seed = side.number_input("Random seed", value=42, step=1)
else:
    side.success(f"Using uploaded file ➜ {Path(uploaded_file.name).name}")

side.markdown("---")
side.header("Highlight settings")

max_k = 30  # hard upper‑bound
_top_k = side.slider("Most‑persistent peaks to highlight", 1, max_k, 4)
side.caption("Plots & table will recolour / resize automatically.")

# ─────────────────────────────────────────────────────────────────────────────
# 🛠️ Advanced detection settings
# ─────────────────────────────────────────────────────────────────────────────

adv = side.expander("Advanced detection settings", expanded=False)
adv.caption("Define the hyper‑parameter *grid* the algorithm will sweep.")

# Layout helper – each parameter gets 3 number_inputs (min/max/steps)

def triple(label, default):
    c1, c2, c3 = adv.columns(3)
    lo = c1.number_input(f"{label} min", value=default[0])
    hi = c2.number_input(f"{label} max", value=default[1])
    stp = c3.number_input(f"{label} steps", min_value=1, value=default[2])
    return lo, hi, stp

params_defaults = {
    "Smoothing": (0, 5, 3),  # ints
    "Bins factor": (1, 2, 2),  # ints
    "Threshold": (0.0, 0.15, 10),
    "Width": (1.0, 50.0, 10),
    "Prominence": (0.01, 1.0, 10),
    "Distance": (1, 20, 5),  # ints
}

(
    (smooth_min, smooth_max, smooth_n),
    (bins_min, bins_max, bins_n),
    (thr_min, thr_max, thr_n),
    (wid_min, wid_max, wid_n),
    (prom_min, prom_max, prom_n),
    (dist_min, dist_max, dist_n),
) = [triple(k, v) for k, v in params_defaults.items()]

merging_range = st.sidebar.slider(
    "Merging range (channels)",
    min_value=0,
    max_value=100,
    value=10,
    help="Cluster peaks whose original‐channel indices are within this many channels of each other"
)

run_btn = side.button("Run analysis", type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# 🚀 Generate / load spectrum & run detection
# ─────────────────────────────────────────────────────────────────────────────

if "init" not in st.session_state or run_btn:

    if uploaded_file is None:
        # -- generate synthetic --
        rng = np.random.default_rng(int(seed))
        x = np.linspace(0, 1000, 1000)
        spectrum = np.zeros_like(x)
        centres = rng.uniform(50, 950, size=n_peaks)
        widths = rng.uniform(8, 25, size=n_peaks)
        amplitudes = rng.uniform(0.6, 1.0, size=n_peaks)
        for c, w, a in zip(centres, widths, amplitudes):
            spectrum += a * np.exp(-((x - c) / w) ** 2)
        spectrum += noise_sigma * rng.standard_normal(len(x))
    else:
        try:
            x, spectrum = load_txt_signal(uploaded_file)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

    # -- build parameter ranges from sidebar --
    smoothing_range = build_linspace(smooth_min, smooth_max, int(smooth_n), as_int=True)
    bins_factor_range = build_linspace(bins_min, bins_max, int(bins_n), as_int=True)
    threshold_range = build_linspace(thr_min, thr_max, int(thr_n))
    width_range = build_linspace(wid_min, wid_max, int(wid_n))
    prominence_range = build_linspace(prom_min, prom_max, int(prom_n))
    distance_range = np.array(build_linspace(dist_min, dist_max, int(dist_n), as_int=True))

    # Safeguard top_k vs expected max
    top_k = min(_top_k, max_k)

    # -- run detection --
    with st.spinner("Running hyper‑grid peak detection…"):
        peaks_info = find_peaks_volumetric_persistence(
            spectrum,
            smoothing_range,
            bins_factor_range,
            threshold_range,
            width_range,
            prominence_range,
            distance_range,
            merging_range=merging_range,
            tol=1,
            parallel=True,
            top_k=max_k,  # compute ample; we slice later
        )

    # cache in session_state
    st.session_state.update(
        {
            "x": x,
            "spectrum": spectrum,
            "peaks_info": peaks_info,
            "top_k": top_k,
        }
    )
    st.session_state["init"] = True

# ---------------- Pull from state ----------------

x = st.session_state["x"]
spectrum = st.session_state["spectrum"]
peaks_info = st.session_state["peaks_info"]
top_k = st.session_state["top_k"]

# ---------------- dynamic colour palette ----------------

base_colours = ["#C94040", "#69995D", "#CBAC88", "#394648"]
colour_cycle = (base_colours * ceil(top_k / len(base_colours)))[:top_k]
COLORS = {
    "spectrum": "#8499C4",
    "detected_peak": colour_cycle,
    "alphas": [1.0] * len(colour_cycle),
    "linewidths": [2.0] * len(colour_cycle),
}

# ─────────────────────────────────────────────────────────────────────────────
# 📈 Plots
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
### 1️⃣ Spectrum & highlighted peaks
Shows the generated (or uploaded) spectrum in blue and the **top‑k** peaks ranked by volumetric persistence. Each coloured dashed line marks one peak’s channel.
""")

plot_spectrum_with_detected_peaks(
    x, spectrum, peaks_info, top_k=top_k, detected_style="vertical_line", COLORS=COLORS
)
st.pyplot(plt.gcf())
plt.clf()

st.markdown("""
### 2️⃣ Persistence barcode
Each vertical bar represents a detected peak; the bar height equals its integrated volumetric persistence across the hyper‑parameter grid. Taller bars ⇒ more robust peaks.
""")

barcode_cols = {
    "default": "#8499C4",
    "detected_peak": colour_cycle,
    "alphas": [1.0] * (len(colour_cycle) + 1),
    "linewidths": [2.0] * (len(colour_cycle) + 1),
}
plot_volumetric_persistence_barcode(peaks_info, COLORS=barcode_cols)
st.pyplot(plt.gcf())
plt.clf()

st.markdown("""
### 3️⃣ Inner‑grid exploration (first peak)
Dive into the hyper‑parameter combinations that detect the **strongest** peak. Here we fix `distance` = 10 and plot the remaining free dimensions (Width - Threshold - Prominence) as a point cloud; marker size ⇢ count of detections.
""")

plot_candidate_inner_grid(
    peaks_info[0],
    outer_smoothing=peaks_info[0]["grid_params"][0]["smoothing"],
    outer_bins=peaks_info[0]["grid_params"][0]["bins_factor"],
    fixed_params={"distance": 10},
    view_type="points",
    single_color=colour_cycle[0],
)

st.markdown("""
### 4️⃣ Radar chart – parameter‑space coverage
The polygon shows how broadly each peak spans the explored range of *each* hyper‑parameter (normalised 0‑1). Peaks covering larger areas survive more settings and are therefore more persistent.
""")

EXPLORED = {
    "smoothing": (smooth_min, smooth_max),
    "bins_factor": (bins_min, bins_max),
    "threshold": (thr_min, thr_max),
    "width": (wid_min, wid_max),
    "prominence": (prom_min, prom_max),
    "distance": (dist_min, dist_max),
}

plot_multi_volumetric_persistence_radar_polygon(
    peaks_info[:top_k],
    explored_ranges=EXPLORED,
    colors=colour_cycle,
    labels=[f"Peak {p['peak_index']}" for p in peaks_info[:top_k]],
    title="",
)
st.pyplot(plt.gcf())
plt.clf()

# ─────────────────────────────────────────────────────────────────────────────
# 🗃️ Data table
# ─────────────────────────────────────────────────────────────────────────────

table = pd.DataFrame(
    {
        "Rank": list(range(1, top_k + 1)),
        "Channel": [p["peak_index"] for p in peaks_info[:top_k]],
        "Persistence": [p["persistence"] for p in peaks_info[:top_k]],
    }
)

st.markdown("""
### 📋 Top‑k peak details
Interactive table – click column headers to sort.
""")

st.dataframe(table, hide_index=True)
