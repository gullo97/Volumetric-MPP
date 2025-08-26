# Volumetric Multi-Parameter Persistence (VM-PP) for Peak Detection

This repository contains the implementation of the **Volumetric Multi-Parameter Persistence (VM-PP)** approach for robust peak detection in noisy signals, as detailed in the paper *"A Multi-Parameter Persistence Algorithm for the Automatic Energy Calibration of Scintillating Radiation Sensors"* by G. Ferranti et al., available as a preprint at https://www.preprints.org/manuscript/202506.0487/v1.

## 🎯 Purpose

The VM-PP algorithm addresses the challenging problem of **automated peak detection in noisy 1D signals**, particularly for energy calibration of scintillating radiation sensors. Traditional peak detection methods often struggle with:

- **Parameter sensitivity**: Small changes in detection parameters can dramatically affect results
- **Noise robustness**: Distinguishing true peaks from noise artifacts
- **Reproducibility**: Ensuring consistent results across different signal conditions

VM-PP solves these issues by exploring a **hyperparameter space** and quantifying each detected peak's **volumetric persistence** - a measure of how consistently a peak appears across different parameter combinations.

## 🔬 How It Works

The algorithm operates in three main phases:

### 1. **Outer Grid Processing**
- **Smoothing**: Applies moving average filters with different window sizes
- **Aggregation**: Bins channels with various aggregation factors to reduce noise

### 2. **Inner Grid Search**
- For each outer grid combination, performs peak detection across a 4D parameter space:
  - **Threshold**: Minimum peak height
  - **Width**: Expected peak width range
  - **Prominence**: Peak prominence above surrounding baseline
  - **Distance**: Minimum separation between peaks

### 3. **Persistence Calculation**
- Counts how many parameter combinations detect each peak
- Merges nearby detections across different preprocessing settings
- Ranks peaks by their **volumetric persistence** (detection frequency)

## 📁 Repository Structure

```
├── core.py                    # Main VM-PP algorithm implementation
├── utils.py                   # Visualization and utility functions
├── usage.ipynb               # Jupyter notebook with examples
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── Data/                     # Sample datasets
└── old_code/                 # Previous iterations and experiments
```

## 🚀 Quick Start

### Installation

```bash
git clone <repository-url>
cd "Volumetric MPP"
pip install -r requirements.txt
```

### Basic Usage

```python
import numpy as np
from core import find_peaks_volumetric_persistence
from utils import plot_spectrum_with_detected_peaks, plot_volumetric_persistence_barcode

# Generate or load your 1D spectrum
x = np.linspace(0, 1000, 1000)
spectrum = your_signal_data  # Replace with actual data

# Define parameter ranges for the grid search
smoothing_range = [0, 3, 5]
bins_factor_range = [1, 2]
threshold_range = np.linspace(0, 0.15, 10)
width_range = np.linspace(1, 50, 10)
prominence_range = np.linspace(0.01, 1.0, 10)
distance_range = np.array([1, 5, 10, 15, 20])

# Run VM-PP peak detection
peaks_info = find_peaks_volumetric_persistence(
    spectrum,
    smoothing_range=smoothing_range,
    bins_factor_range=bins_factor_range,
    threshold_range=threshold_range,
    width_range=width_range,
    prominence_range=prominence_range,
    distance_range=distance_range,
    merging_range=10,
    tol=1,
    parallel=True,
    top_k=10
)

# Visualize results
plot_spectrum_with_detected_peaks(x, spectrum, peaks_info, top_k=4)
plot_volumetric_persistence_barcode(peaks_info)
```

## 📊 Output Format

The algorithm returns a list of dictionaries, each containing:

- **`peak_index`**: Position of the peak in the original spectrum
- **`persistence`**: Volumetric persistence score (higher = more robust)
- **`grid_params`**: List of all parameter combinations that detected this peak

Peaks are automatically sorted by persistence in descending order, with the most robust peaks listed first.

## 🎮 Interactive Demo

An interactive web demo is available at **https://volumetric-mpp.streamlit.app/** where you can:

- Upload your own signal data (`.txt` files)
- Generate synthetic test signals
- Adjust all hyperparameters in real-time
- Visualize results with multiple plot types
- Export detection results

To run the demo locally:

```bash
streamlit run app.py
```

## 📈 Visualization Options

The `utils.py` module provides several visualization functions:

- **`plot_spectrum_with_detected_peaks`**: Overlay detected peaks on the original spectrum
- **`plot_volumetric_persistence_barcode`**: Barcode plot showing peak persistence scores
- **`plot_candidate_inner_grid`**: 3D visualization of parameter space exploration
- **`plot_multi_volumetric_persistence_radar_polygon`**: Radar plot comparing multiple peaks

## ⚙️ Key Parameters

### Core Algorithm Parameters
- **`merging_range`**: Distance threshold for merging nearby peak detections
- **`tol`**: Tolerance for clustering detections within parameter combinations
- **`parallel`**: Enable multiprocessing for faster computation
- **`top_k`**: Limit output to top K most persistent peaks

### Detection Parameters (Grid Search Ranges)
- **`smoothing_range`**: Moving average window sizes
- **`bins_factor_range`**: Channel aggregation factors
- **`threshold_range`**: Peak height thresholds
- **`width_range`**: Expected peak width ranges
- **`prominence_range`**: Peak prominence requirements
- **`distance_range`**: Minimum peak separation distances

## 🎯 Applications

VM-PP is particularly effective for:

- **Radiation spectroscopy**: Energy calibration of gamma-ray detectors
- **Chromatography**: Peak detection in noisy chromatographic data
- **Time series analysis**: Identifying significant events in temporal data
- **Signal processing**: Robust feature extraction from 1D signals

## 📚 Dependencies

- **NumPy**: Numerical computations
- **SciPy**: Signal processing (`find_peaks` function)
- **Matplotlib**: Plotting and visualization
- **Streamlit**: Web application framework
- **Pandas**: Data handling (for Streamlit app)

## 🤝 Contributing

This implementation is based on ongoing research. For questions, suggestions, or collaborations, please refer to the associated research paper or contact the authors.

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{ferranti2025mpp,
  title={A Multi-Parameter Persistence Algorithm for the Automatic Energy Calibration of Scintillating Radiation Sensors},
  author={Ferranti, G. and others},
  year={2025},
  journal={Sensors, 25, 4579},
  doi={https://doi.org/10.3390/s25154579}
}
```

---

**Interactive Demo**: https://volumetric-mpp.streamlit.app/ 
