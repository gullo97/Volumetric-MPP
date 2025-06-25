"""
Calibration Package for Volumetric MPP

This package provides comprehensive detector calibration functionality using
radioactive sources and volumetric persistence peak detection.

Main modules:
- calibration_core: Core calibration functions
- calibration_plots: Visualization and plotting functions  
- source_config: Radioactive source configurations

Quick start:
    from calibration import get_source_config, process_source_data, calibrate_detector_array
"""

from .calibration_core import (
    process_source_data,
    calibrate_detector_array,
    find_poor_detectors,
    channel_to_energy,
    energy_to_channel,
    create_parameter_grid,
    print_parameter_grid,
    get_optimized_parameter_suggestions,
    create_custom_detection_params
)

from .calibration_plots import (
    plot_calibration_quality_overview,
    plot_single_detector_calibration,
    plot_combined_source_spectra,
    plot_3d_energy_spectrum,
    plot_2d_energy_heatmap
)

from .source_config import (
    get_source_config,
    get_detection_params,
    list_available_sources,
    print_source_info,
    RADIOACTIVE_SOURCES
)

__version__ = "1.0.0"
__author__ = "Volumetric MPP Team"

__all__ = [
    # Core functions
    'process_source_data',
    'calibrate_detector_array',
    'find_poor_detectors',
    'channel_to_energy',
    'energy_to_channel',
    
    # Hyperparameter functions
    'create_parameter_grid',
    'print_parameter_grid',
    'get_optimized_parameter_suggestions',
    'create_custom_detection_params',
    
    # Plotting functions
    'plot_calibration_quality_overview',
    'plot_single_detector_calibration',
    'plot_combined_source_spectra',
    'plot_3d_energy_spectrum',
    'plot_2d_energy_heatmap',
    
    # Configuration functions
    'get_source_config',
    'get_detection_params',
    'list_available_sources',
    'print_source_info',
    
    # Constants
    'RADIOACTIVE_SOURCES'
]
