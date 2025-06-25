"""
Configuration file for radioactive sources used in detector calibration.

This module contains predefined configurations for common radioactive sources
including their expected emission energies and detection parameters.
"""

# Predefined radioactive sources with their gamma-ray emission energies (in MeV)
RADIOACTIVE_SOURCES = {
    "Sodium": {
        "energies": [0.511],  # Na-22: 511 keV (positron annihilation)
        "description": "Na-22 - Positron emitter with 511 keV annihilation peak"
    },
    "Cobalt": {
        "energies": [1.17, 1.33],  # Co-60: 1173 keV and 1332 keV
        "description": "Co-60 - Two gamma rays at 1173 keV and 1332 keV"
    },
    "Cesium": {
        "energies": [0.662],  # Cs-137: 662 keV
        "description": "Cs-137 - Single gamma ray at 662 keV"
    },
    "Americium": {
        "energies": [0.0595],  # Am-241: 59.5 keV
        "description": "Am-241 - Low energy gamma ray at 59.5 keV"
    },
    "Barium": {
        "energies": [0.081, 0.356],  # Ba-133: 81 keV and 356 keV (main peaks)
        "description": "Ba-133 - Multiple gamma rays, main peaks at 81 and 356 keV"
    },
    "Europium": {
        "energies": [0.122, 0.244, 0.344, 0.779, 0.964, 1.408],  # Eu-152
        "description": "Eu-152 - Multiple gamma rays for energy calibration"
    },
    "Manganese": {
        "energies": [0.835],  # Mn-54: 835 keV
        "description": "Mn-54 - Single gamma ray at 835 keV"
    }
}

# Default detection parameters for volumetric persistence
DEFAULT_DETECTION_PARAMS = {
    "smoothing_range": [1, 3, 5],
    "bins_factor_range": [1, 2],
    "threshold_range": [0.01, 0.05, 0.1],
    "width_range": [1, 3, 5],
    "prominence_range": [0.1, 0.3, 0.5],
    "distance_range": [5, 10, 15],
    "merging_range": 5,
    "tol": 1,
    "parallel": True
}

# Optimized detection parameters for specific source types
OPTIMIZED_DETECTION_PARAMS = {
    "low_energy": {  # For sources like Am-241
        "smoothing_range": [1, 2],
        "bins_factor_range": [1],
        "threshold_range": [0.01, 0.03],
        "width_range": [1, 2, 3],
        "prominence_range": [0.05, 0.1, 0.2],
        "distance_range": [3, 5, 8],
        "merging_range": 3,
        "tol": 1,
        "parallel": True
    },
    "medium_energy": {  # For sources like Cs-137, Na-22
        "smoothing_range": [1, 3, 5],
        "bins_factor_range": [1, 2],
        "threshold_range": [0.01, 0.05, 0.1],
        "width_range": [2, 4, 6],
        "prominence_range": [0.1, 0.3, 0.5],
        "distance_range": [5, 10, 15],
        "merging_range": 5,
        "tol": 1,
        "parallel": True
    },
    "high_energy": {  # For sources like Co-60
        "smoothing_range": [3, 5, 8],
        "bins_factor_range": [1, 2, 3],
        "threshold_range": [0.05, 0.1, 0.2],
        "width_range": [3, 5, 8],
        "prominence_range": [0.2, 0.4, 0.6],
        "distance_range": [8, 15, 20],
        "merging_range": 8,
        "tol": 2,
        "parallel": True
    },
    "multi_peak": {  # For sources with multiple peaks like Eu-152
        "smoothing_range": [1, 3, 5],
        "bins_factor_range": [1, 2],
        "threshold_range": [0.01, 0.03, 0.05],
        "width_range": [1, 3, 5, 8],
        "prominence_range": [0.05, 0.1, 0.2, 0.3],
        "distance_range": [5, 10, 15, 20],
        "merging_range": 5,
        "tol": 1,
        "parallel": True
    }
}

# Recommended top_k values for different sources
RECOMMENDED_TOP_K = {
    "Sodium": 2,      # Usually see 1-2 peaks
    "Cobalt": 5,      # Usually see 2 main peaks + potential Compton edges
    "Cesium": 2,      # Usually see 1-2 peaks
    "Americium": 1,   # Low energy, usually 1 peak
    "Barium": 3,      # Multiple peaks but focus on main ones
    "Europium": 8,    # Many peaks for calibration
    "Manganese": 2    # Usually 1-2 peaks
}

# Source energy categories for automatic parameter selection
SOURCE_ENERGY_CATEGORIES = {
    "low_energy": ["Americium"],
    "medium_energy": ["Sodium", "Cesium", "Barium"],
    "high_energy": ["Cobalt", "Manganese"],
    "multi_peak": ["Europium", "Barium"]
}


def get_source_config(source_name: str) -> dict:
    """
    Get complete configuration for a radioactive source.
    
    Parameters:
        source_name (str): Name of the radioactive source
        
    Returns:
        dict: Complete source configuration
    """
    if source_name not in RADIOACTIVE_SOURCES:
        raise ValueError(f"Unknown source: {source_name}. "
                         f"Available sources: {list(RADIOACTIVE_SOURCES.keys())}")
    
    source_info = RADIOACTIVE_SOURCES[source_name]
    
    # Determine energy category
    energy_category = "medium_energy"  # default
    for category, sources in SOURCE_ENERGY_CATEGORIES.items():
        if source_name in sources:
            energy_category = category
            break
    
    return {
        "name": source_name,
        "energies": source_info["energies"],
        "description": source_info["description"],
        "energy_category": energy_category,
        "detection_params": OPTIMIZED_DETECTION_PARAMS[energy_category],
        "top_k": RECOMMENDED_TOP_K.get(source_name, 3)
    }


def get_detection_params(energy_category: str = "medium_energy") -> dict:
    """
    Get detection parameters for a specific energy category.
    
    Parameters:
        energy_category (str): Energy category 
                              ('low_energy', 'medium_energy', 'high_energy', 'multi_peak')
        
    Returns:
        dict: Detection parameters
    """
    if energy_category not in OPTIMIZED_DETECTION_PARAMS:
        print(f"Warning: Unknown energy category '{energy_category}'. "
              f"Using default parameters.")
        return DEFAULT_DETECTION_PARAMS
    
    return OPTIMIZED_DETECTION_PARAMS[energy_category]


def list_available_sources() -> list:
    """
    Get list of all available radioactive sources.
    
    Returns:
        list: List of source names
    """
    return list(RADIOACTIVE_SOURCES.keys())


def print_source_info(source_name: str = None):
    """
    Print information about available sources.
    
    Parameters:
        source_name (str, optional): Specific source to show info for.
                                    If None, shows all sources.
    """
    if source_name:
        if source_name in RADIOACTIVE_SOURCES:
            config = get_source_config(source_name)
            print(f"\n{source_name}:")
            print(f"  Description: {config['description']}")
            print(f"  Energies: {config['energies']} MeV")
            print(f"  Energy Category: {config['energy_category']}")
            print(f"  Recommended top_k: {config['top_k']}")
        else:
            print(f"Unknown source: {source_name}")
    else:
        print("\nAvailable Radioactive Sources:")
        print("=" * 50)
        for source, info in RADIOACTIVE_SOURCES.items():
            energies = ", ".join([f"{e:.3f}" for e in info["energies"]])
            print(f"{source:12} : {energies} MeV - {info['description']}")
