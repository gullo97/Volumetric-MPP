# Hyper‑Volumetric Persistence – Streamlit demo (updated)
# ======================================================
# This app lets users generate a synthetic 1‑D spectrum, run the
# Hyper‑Volumetric‑Persistence peak finder (core.py), and interactively
# explore the results.
#
# NEW IN THIS REVISION
# --------------------
# • **top‑k slider** – choose how many of the most‑persistent peaks to
#   highlight/colour.
# • Colours now cycle automatically even when you ask for more peaks
#   than the base palette size.
# • Patched `utils.plot_candidate_inner_grid` to avoid the
#   ``TypeError: unsupported operand type(s) for -: 'list' and 'float'``
#   when more than four peaks are requested (issue was the internal
#   ``edges`` helper assuming a NumPy array).
#
# NOTE: keep this file in the same folder as **core.py** and **utils.py**.

from __future__ import annotations

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import ceil

from core import find_peaks_volumetric_persistence
import utils  # we will monkey‑patch one function below
from utils import (
    plot_spectrum_with_detected_peaks,
    plot_volumetric_persistence_barcode,
    plot_multi_volumetric_persistence_radar_polygon,
)

# -----------------------------------------------------------------------------
# 🔧 Patch utils.plot_candidate_inner_grid to fix list‑minus‑float bug
# -----------------------------------------------------------------------------

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
    counts = Counter(tuple(p[f] for f in free_params) for p in filtered)
    coords = list(counts.keys())
    freqs = list(counts.values())

    # ---------------- plotting helpers ----------------
    def _scatter(ax, xs, ys, zs=None):
        sizes = [f * 40 for f in freqs]
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
            x, y, z = zip(*coords)
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111, projection="3d")
            _scatter(ax, x, y, z)
            ax.set_xlabel(free_params[0])
            ax.set_ylabel(free_params[1])
            ax.set_zlabel(free_params[2])
        elif dim == 2:
            x, y = zip(*coords)
            fig, ax = plt.subplots(figsize=(7, 4))
            _scatter(ax, x, y)
            ax.set_xlabel(free_params[0])
            ax.set_ylabel(free_params[1])
        else:  # 1‑D
            x = [c[0] for c in coords]
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

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

st.title("Hyper‑Volumetric Persistence – Interactive demo")

# ---------------- sidebar controls ----------------
side = st.sidebar
side.header("Signal & detection controls")

n_peaks = side.slider("Number of synthetic peaks", min_value=1, max_value=10, value=3)
noise_sigma = side.slider("Gaussian noise σ", 0.0, 0.2, 0.05, step=0.005)
seed = side.number_input("Random seed", value=42, step=1)

side.markdown("---")

top_k = side.slider("Most‑persistent peaks to highlight", min_value=1, max_value=20, value=4)
side.caption("Plots & table will recolour / resize automatically.")

if side.button("Generate / analyse") or "spectrum" not in st.session_state:
    rng = np.random.default_rng(int(seed))
    x = np.linspace(0, 1000, 1000)

    # ----- build synthetic spectrum -----
    spectrum = np.zeros_like(x)
    centres = rng.uniform(50, 950, size=n_peaks)
    widths = rng.uniform(8, 25, size=n_peaks)
    amplitudes = rng.uniform(0.6, 1.0, size=n_peaks)
    for c, w, a in zip(centres, widths, amplitudes):
        spectrum += a * np.exp(-((x - c) / w) ** 2)
    spectrum += noise_sigma * rng.standard_normal(len(x))

    st.session_state["x"] = x
    st.session_state["spectrum"] = spectrum

    # ----- parameter ranges (same as paper/demo) -----
    smoothing_range = [0, 3, 5]
    bins_factor_range = [1, 2]
    threshold_range = np.linspace(0, 0.15, 10)
    width_range = np.linspace(1, 50, 10)
    prominence_range = np.linspace(0.01, 1.0, 10)
    distance_range = np.array([1, 5, 10, 15, 20])

    peaks_info = find_peaks_volumetric_persistence(
        spectrum,
        smoothing_range,
        bins_factor_range,
        threshold_range,
        width_range,
        prominence_range,
        distance_range,
        merging_range=10,
        tol=1,
        parallel=True,
        top_k=30,
    )
    st.session_state["peaks_info"] = peaks_info

# ---------------- retrieve cached data ----------------

x = st.session_state["x"]
spectrum = st.session_state["spectrum"]
peaks_info = st.session_state["peaks_info"]

# ---------------- colour palette (auto‑extend) ----------------
base_det_colours = ["#C94040", "#69995D", "#CBAC88", "#394648"]
factor = ceil(top_k / len(base_det_colours))
colour_cycle = (base_det_colours * factor)[:top_k]
COLORS = {
    "spectrum": "#8499C4",
    "detected_peak": colour_cycle,
    "alphas": [1.0] * len(colour_cycle),
    "linewidths": [2.0] * len(colour_cycle),
}

# -----------------------------------------------------------------------------
# PLOTS
# -----------------------------------------------------------------------------

# --- spectrum with highlighted peaks ---
st.subheader("Synthetic spectrum & highlighted peaks")
plot_spectrum_with_detected_peaks(
    x,
    spectrum,
    peaks_info,
    top_k=top_k,
    detected_style="vertical_line",
    COLORS=COLORS,
)
st.pyplot(plt.gcf())
plt.clf()

# --- persistence barcode ---
st.subheader("Volumetric‑persistence barcode")
# Extend colour list so the first *top_k* bars get unique colours
barcode_cols = {
    "default": "#8499C4",
    "detected_peak": colour_cycle,
    "alphas": [1.0] * (len(colour_cycle) + 1),
    "linewidths": [2.0] * (len(colour_cycle) + 1),
}
plot_volumetric_persistence_barcode(peaks_info, COLORS=barcode_cols)
st.pyplot(plt.gcf())
plt.clf()

# --- inner‑grid exploration for first peak ---
st.subheader("Inner‑grid exploration (first peak)")
plot_candidate_inner_grid(
    peaks_info[0],
    outer_smoothing=peaks_info[0]["grid_params"][0]["smoothing"],
    outer_bins=peaks_info[0]["grid_params"][0]["bins_factor"],
    fixed_params={"distance": 10},
    view_type="points",  # safer than 'voxels' (buggy in some cases)
    single_color=colour_cycle[0],
)

# --- radar/polygon plot ---
EXPLORED = {
    "smoothing": (0, 11),
    "bins_factor": (1, 2.5),
    "threshold": (0.0, 0.2),
    "width": (1.0, 50.0),
    "prominence": (0.01, 1.0),
    "distance": (1, 25),
}

st.subheader("Persistence‑coverage radar")
plot_multi_volumetric_persistence_radar_polygon(
    peaks_info[:top_k],
    explored_ranges=EXPLORED,
    colors=colour_cycle,
    labels=[f"Peak {p['peak_index']}" for p in peaks_info[:top_k]],
    title="",
)
st.pyplot(plt.gcf())
plt.clf()

# -----------------------------------------------------------------------------
# Data table
# -----------------------------------------------------------------------------

table = pd.DataFrame(
    {
        "Rank": list(range(1, top_k + 1)),
        "Channel": [p["peak_index"] for p in peaks_info[:top_k]],
        "Persistence": [p["persistence"] for p in peaks_info[:top_k]],
    }
)

st.subheader("Top‑k persistent peaks – details")
st.dataframe(table, hide_index=True)
