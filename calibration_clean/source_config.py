"""
Configuration file for radioactive sources used in detector calibration.

This module contains predefined configurations for common radioactive sources
including their expected emission energies.
"""
import numpy as np

# Predefined radioactive sources with their gamma-ray emission energies (in MeV)
RADIOACTIVE_SOURCES = {
    "Sodium": {
        "energies": [0.511],  # Na-22: 511 keV (positron annihilation)
        "description": "Na-22 - Positron emitter with 511 keV annihilation peak",
        "top_k": 1  # Only 1 peak expected with channel ranges
    },
    "Cobalt": {
        "energies": [1.17, 1.33],  # Co-60: 1173 keV and 1332 keV
        "description": "Co-60 - Two gamma rays at 1173 keV and 1332 keV",
        "top_k": 2  # Exactly 2 peaks expected with channel ranges
    },
    "Cesium": {
        "energies": [0.662],  # Cs-137: 662 keV
        "description": "Cs-137 - Single gamma ray at 662 keV",
        "top_k": 1  # Only 1 peak expected with channel ranges
    },
    "Americium": {
        "energies": [0.0595],  # Am-241: 59.5 keV
        "description": "Am-241 - Low energy gamma ray at 59.5 keV",
        "top_k": 1
    },
    "Barium": {
        "energies": [0.081, 0.356],  # Ba-133: 81 keV and 356 keV (main peaks)
        "description": "Ba-133 - Multiple gamma rays, main peaks at 81 and 356 keV",
        "top_k": 2
    },
    "Europium": {
        "energies": [0.122, 0.244, 0.344, 0.779, 0.964, 1.408],  # Eu-152
        "description": "Eu-152 - Multiple gamma rays for energy calibration",
        "top_k": 6
    },
    "Manganese": {
        "energies": [0.835],  # Mn-54: 835 keV
        "description": "Mn-54 - Single gamma ray at 835 keV",
        "top_k": 1
    }
}

# Global detection parameters for volumetric persistence
# These are optimized for AUC-normalized spectra (values 0-1)
GLOBAL_DETECTION_PARAMS = {
    "smoothing_range": [3, 5, 7],
    "bins_factor_range": [1, 2],
    "threshold_range": [float(x) for x in np.linspace(0.00001, 0.02, 8)],
    "width_range": [float(x) for x in np.linspace(1, 50, 7)],
    "prominence_range": [float(x) for x in np.linspace(0.0001, 0.01, 9)],
    "distance_range": [1, 5, 10, 15],
    "merging_range": 5,
    "tol": 1,
    "parallel": True
}

# Expected channel ranges for peak validation (adjust for your detector setup)
EXPECTED_CHANNEL_RANGES = {
    "Sodium": [(400, 700)],      # 511 keV
    "Cesium": [(500, 900)],      # 662 keV
    "Cobalt": [(1050, 1400)],  # 1173 & 1332 keV
    "Americium": [(50, 150)],    # 59.5 keV
    "Barium": [(70, 120), (300, 400)],      # 81 & 356 keV
    "Europium": [(100, 200), (200, 300), (300, 400), (700, 850), (900, 1050), (1300, 1500)],
    "Manganese": [(750, 950)]    # 835 keV
}


def get_source_config(source_name: str) -> dict:
    """Get configuration for a specific radioactive source."""
    if source_name not in RADIOACTIVE_SOURCES:
        raise ValueError(f"Unknown source: {source_name}. Available: {list(RADIOACTIVE_SOURCES.keys())}")
    
    config = RADIOACTIVE_SOURCES[source_name].copy()
    config['channel_ranges'] = EXPECTED_CHANNEL_RANGES.get(source_name, [])
    return config


def print_source_info():
    """Print information about all available sources."""
    for name, config in RADIOACTIVE_SOURCES.items():
        energies_str = ', '.join(f"{e:.3f}" for e in config['energies'])
        print(f"  {name}: {energies_str} MeV - {config['description']}")


def list_available_sources() -> list:
    """Return list of available source names."""
    return list(RADIOACTIVE_SOURCES.keys())
