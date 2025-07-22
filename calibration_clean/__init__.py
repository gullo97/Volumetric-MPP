"""
Clean Calibration Module

This module provides a streamlined version of the detector calibration system
without the complex parameter optimization that was causing issues.

Key features:
- Simple global parameter configuration
- Working volumetric persistence peak detection
- Complete visualization capabilities
- Robust calibration pipeline

Modules:
- source_config: Source definitions and global parameters
- calibration_core: Core calibration functions
- calibration_plots: All visualization functions
- calibration_notebook.ipynb: Clean, working notebook
"""

from .source_config import (
    get_source_config,
    print_source_info,
    list_available_sources,
    GLOBAL_DETECTION_PARAMS
)

from .calibration_core import (
    process_source_data,
    calibrate_detector_array,
    find_poor_detectors,
    channel_to_energy,
    energy_to_channel
)

from .calibration_plots import (
    plot_calibration_quality_overview,
    plot_single_detector_calibration,
    plot_combined_source_spectra,
    plot_3d_energy_spectrum,
    plot_2d_energy_heatmap
)

__all__ = [
    'get_source_config',
    'print_source_info', 
    'list_available_sources',
    'GLOBAL_DETECTION_PARAMS',
    'process_source_data',
    'calibrate_detector_array',
    'find_poor_detectors',
    'channel_to_energy',
    'energy_to_channel',
    'plot_calibration_quality_overview',
    'plot_single_detector_calibration',
    'plot_combined_source_spectra',
    'plot_3d_energy_spectrum',
    'plot_2d_energy_heatmap'
]
