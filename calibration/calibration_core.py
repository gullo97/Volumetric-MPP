"""
Calibration Core Functions for Volumetric MPP

This module provides functions for detector calibration using radioactive sources
and the new volumetric persistence peak detection method.
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress
from typing import Dict, List, Tuple, Union
import sys
import os

# Add parent directory to path to import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import find_peaks_volumetric_persistence


def load_data_from_excel(file_path: str, sheet_name: str) -> np.ndarray:
    """
    Load spectrum data from Excel sheet.
    
    Parameters:
        file_path (str): Path to Excel file
        sheet_name (str): Name of the sheet to read
        
    Returns:
        np.ndarray: Spectrum data with shape (n_channels, n_detectors)
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    data_numeric = df.apply(pd.to_numeric, errors='coerce').dropna(how='all')
    if data_numeric.empty:
        raise ValueError(f"No numeric data found in sheet {sheet_name}.")
    return data_numeric.values


def load_data_from_csv(file_path: str) -> np.ndarray:
    """
    Load spectrum data from CSV file.
    
    Parameters:
        file_path (str): Path to CSV file
        
    Returns:
        np.ndarray: Spectrum data with shape (n_channels, n_detectors)
    """
    df = pd.read_csv(file_path)
    data_numeric = df.apply(pd.to_numeric, errors='coerce').dropna(how='all')
    if data_numeric.empty:
        raise ValueError(f"No numeric data found in CSV file {file_path}.")
    return data_numeric.values


def detect_peaks_single_detector(spectrum: np.ndarray, 
                                detection_params: Dict,
                                top_k: int = 5,
                                extended_top_k: int = None
                                ) -> Tuple[List[Dict], List[Dict]]:
    """
    Detect peaks in a single detector spectrum using volumetric persistence.
    
    Parameters:
        spectrum (np.ndarray): 1D spectrum array
        detection_params (Dict): Parameters for peak detection
        top_k (int): Number of top peaks to return for calibration
        extended_top_k (int): Larger number for visualization 
                             (default: top_k * 3)
        
    Returns:
        Tuple[List[Dict], List[Dict]]: (calibration_peaks, all_peaks_for_viz)
            - calibration_peaks: Top-k peaks for calibration
            - all_peaks_for_viz: Extended peaks for visualization
    """
    if extended_top_k is None:
        # At least 3x or 20 peaks for visualization
        extended_top_k = max(top_k * 3, 20)  
    
    # Get extended peaks for visualization
    all_peaks = find_peaks_volumetric_persistence(
        spectrum,
        smoothing_range=detection_params['smoothing_range'],
        bins_factor_range=detection_params['bins_factor_range'],
        threshold_range=detection_params['threshold_range'],
        width_range=detection_params['width_range'],
        prominence_range=detection_params['prominence_range'],
        distance_range=detection_params['distance_range'],
        merging_range=detection_params.get('merging_range', 5),
        tol=detection_params.get('tol', 1),
        parallel=detection_params.get('parallel', True),
        top_k=extended_top_k  # Get more peaks for visualization
    )
    
    # Extract calibration peaks (top-k)
    calibration_peaks = all_peaks[:top_k]
    
    return calibration_peaks, all_peaks


def detect_peaks_multi_detector(spectra_array: np.ndarray,
                               detection_params: Dict,
                               top_k: int = 5) -> Dict[int, Dict]:
    """
    Detect peaks across multiple detectors.
    
    Parameters:
        spectra_array (np.ndarray): Spectrum data with shape 
                                   (n_channels, n_detectors)
        detection_params (Dict): Parameters for peak detection
        top_k (int): Number of top peaks to return per detector
        
    Returns:
        Dict[int, Dict]: Dictionary mapping detector index to:
            {'calibration_peaks': List[Dict], 'all_peaks': List[Dict]}
    """
    n_channels, n_detectors = spectra_array.shape
    results = {}
    
    for detector_idx in range(n_detectors):
        spectrum = spectra_array[:, detector_idx]
        calibration_peaks, all_peaks = detect_peaks_single_detector(
            spectrum, detection_params, top_k
        )
        results[detector_idx] = {
            'calibration_peaks': calibration_peaks,
            'all_peaks': all_peaks
        }
    
    return results


def calibrate_detector(peak_channels: List[float], 
                       energies: List[float]) -> Dict:
    """
    Perform linear calibration for a single detector.
    
    Parameters:
        peak_channels (List[float]): Detected peak channel positions
        energies (List[float]): Corresponding known energies
        
    Returns:
        Dict: Calibration parameters and cross-validation error
    """
    peak_channels = np.array(peak_channels)
    energies = np.array(energies)
    
    if len(peak_channels) < 2:
        return {
            'slope': None,
            'intercept': None,
            'cv_error': None,
            'n_points': len(peak_channels),
            'r_squared': None
        }
    
    # Sort by channel
    sort_idx = np.argsort(peak_channels)
    x = peak_channels[sort_idx]
    y = energies[sort_idx]
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    # Leave-one-out cross-validation
    errors = []
    n = len(x)
    for i in range(n):
        x_train = np.delete(x, i)
        y_train = np.delete(y, i)
        if len(x_train) < 2:
            continue
        s, b, _, _, _ = linregress(x_train, y_train)
        pred = s * x[i] + b
        errors.append(abs(pred - y[i]))
    
    cv_error = np.mean(errors) if errors else None
    
    return {
        'slope': slope,
        'intercept': intercept,
        'cv_error': cv_error,
        'n_points': n,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err
    }


def process_source_data(file_path: str,
                       source_config: Dict,
                       detection_params: Dict) -> Tuple[np.ndarray, Dict]:
    """
    Process data for a single radioactive source.
    
    Parameters:
        file_path (str): Path to data file
        source_config (Dict): Source configuration containing sheet_name, 
                             expected_energies, top_k
        detection_params (Dict): Peak detection parameters
        
    Returns:
        Tuple[np.ndarray, Dict]: Spectrum array and peak detection results
                                per detector containing both calibration
                                and all peaks for visualization
    """
    # Load data based on file extension
    if file_path.endswith('.xlsx'):
        spectra_array = load_data_from_excel(file_path, 
                                            source_config['sheet_name'])
    elif file_path.endswith('.csv'):
        spectra_array = load_data_from_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Detect peaks (now returns both calibration and all peaks)
    peaks_per_detector = detect_peaks_multi_detector(
        spectra_array, 
        detection_params, 
        source_config['top_k']
    )
    
    return spectra_array, peaks_per_detector


def validate_peak_in_range(peak_channel: float, 
                          expected_ranges: List[Tuple[int, int]]) -> bool:
    """
    Check if a peak channel falls within any of the expected ranges.
    
    Parameters:
        peak_channel (float): Detected peak channel position
        expected_ranges (List[Tuple[int, int]]): List of (min, max) channel ranges
        
    Returns:
        bool: True if peak is within any expected range
    """
    for min_ch, max_ch in expected_ranges:
        if min_ch <= peak_channel <= max_ch:
            return True
    return False


def apply_source_specific_peak_selection_robust(persistent_peaks: List[Dict],
                                               source_name: str,
                                               expected_energies: List[float]
                                               ) -> List[float]:
    """
    Apply enhanced source-specific peak selection with channel range validation.
    
    This function implements iterative peak selection that validates peaks
    against expected channel ranges and selects the most persistent valid peaks.
    
    Parameters:
        persistent_peaks (List[Dict]): Persistence-ranked peaks (highest first)
        source_name (str): Name of the radioactive source
        expected_energies (List[float]): Expected energies for this source
        
    Returns:
        List[float]: Selected peak channels for calibration
    """
    from source_config import SOURCE_CHANNEL_RANGES
    
    if not persistent_peaks:
        return []
    
    expected_count = len(expected_energies)
    
    # Get expected channel ranges for this source
    if source_name in SOURCE_CHANNEL_RANGES:
        expected_ranges = SOURCE_CHANNEL_RANGES[source_name]["expected_ranges"]
    else:
        print(f"Warning: No channel ranges defined for {source_name}, "
              f"using fallback selection")
        # Fallback to original method without range validation
        return apply_source_specific_peak_selection(
            persistent_peaks, source_name, expected_energies
        )
    
    # Step 1: Filter peaks by channel ranges
    valid_peaks = []
    print(f"   Debug: Filtering {len(persistent_peaks)} peaks for {source_name}")
    print(f"   Expected ranges: {expected_ranges}")
    
    for i, peak in enumerate(persistent_peaks):
        if validate_peak_in_range(peak['peak_index'], expected_ranges):
            valid_peaks.append(peak)
            print(f"   ✅ Peak {i}: ch={peak['peak_index']:.1f}, "
                  f"pers={peak['persistence']:.3f} - VALID")
        else:
            print(f"   ❌ Peak {i}: ch={peak['peak_index']:.1f}, "
                  f"pers={peak['persistence']:.3f} - outside range")
    
    print(f"   Found {len(valid_peaks)} valid peaks out of {len(persistent_peaks)}")
    
    if not valid_peaks:
        print(f"Warning: No peaks found in expected ranges for {source_name}, "
              f"falling back to standard selection")
        return apply_source_specific_peak_selection(
            persistent_peaks, source_name, expected_energies
        )
    
    # Step 2: Source-specific selection from valid peaks
    # Note: valid_peaks are already sorted by persistence (highest first)
    # from the original persistent_peaks list
    
    if source_name.lower() in ["sodium", "cesium"]:
        # For single-peak sources: select highest persistence peak in range
        if expected_count == 1:
            # Take the first (highest persistence) valid peak
            selected_peak = valid_peaks[0]
            print(f"   Selected for {source_name}: ch={selected_peak['peak_index']:.1f}, "
                  f"pers={selected_peak['persistence']:.3f}")
            return [selected_peak['peak_index']]
        else:
            print(f"Warning: {source_name} expected to have 1 peak, "
                  f"got {expected_count} expected energies")
            # Take the first (highest persistence) valid peak
            selected_peak = valid_peaks[0]
            print(f"   Selected for {source_name}: ch={selected_peak['peak_index']:.1f}, "
                  f"pers={selected_peak['persistence']:.3f}")
            return [selected_peak['peak_index']]
    
    elif source_name.lower() == "cobalt":
        # For Cobalt: need 2 peaks, select by highest persistence in ranges
        if expected_count == 2:
            if len(valid_peaks) >= 2:
                # Take the 2 highest persistence peaks and sort by channel
                selected_peaks = valid_peaks[:2]
                # Sort by channel for proper energy assignment
                selected_peaks.sort(key=lambda x: x['peak_index'])
                channels = [peak['peak_index'] for peak in selected_peaks]
                persistences = [peak['persistence'] for peak in selected_peaks]
                print(f"   Selected for {source_name}: "
                      f"ch=[{channels[0]:.1f}, {channels[1]:.1f}], "
                      f"pers=[{persistences[0]:.3f}, {persistences[1]:.3f}]")
                return channels
            else:
                print(f"Warning: Only {len(valid_peaks)} valid peaks found "
                      f"for Cobalt, expected 2")
                # Use what we have, sorted by channel
                selected_peaks = sorted(valid_peaks, 
                                      key=lambda x: x['peak_index'])
                channels = [peak['peak_index'] for peak in selected_peaks]
                print(f"   Selected for {source_name} (limited): {channels}")
                return channels
        else:
            print(f"Warning: Cobalt expected to have 2 peaks, "
                  f"got {expected_count} expected energies")
            # Take the highest persistence peaks up to expected count
            selected_peaks = valid_peaks[:expected_count]
            selected_peaks.sort(key=lambda x: x['peak_index'])
            channels = [peak['peak_index'] for peak in selected_peaks]
            print(f"   Selected for {source_name} (adjusted): {channels}")
            return channels
    
    else:
        # Default behavior: select highest persistence peaks within range
        # and sort by channel for proper energy assignment
        selected_peaks = valid_peaks[:expected_count]
        selected_peaks.sort(key=lambda x: x['peak_index'])
        channels = [peak['peak_index'] for peak in selected_peaks]
        print(f"   Selected for {source_name} (default): {channels}")
        return channels


def apply_source_specific_peak_selection(persistent_peaks: List[Dict],
                                         source_name: str,
                                         expected_energies: List[float]
                                         ) -> List[float]:
    """
    Apply source-specific peak selection rules to choose calibration peaks
    from the most persistent peaks.

    Parameters:
        persistent_peaks (List[Dict]): Top-K most persistent peaks
        source_name (str): Name of the radioactive source
        expected_energies (List[float]): Expected energies for this source

    Returns:
        List[float]: Selected peak channels for calibration
    """
    if not persistent_peaks:
        return []
    
    expected_count = len(expected_energies)
    
    # Source-specific selection rules
    if source_name.lower() in ["sodium", "cesium"]:
        # For Sodium and Cesium: select peak at highest channel
        if expected_count == 1:
            # Select the peak with highest channel position
            selected_peak = max(persistent_peaks,
                                key=lambda x: x['peak_index'])
            return [selected_peak['peak_index']]
        else:
            print(f"Warning: {source_name} expected to have 1 peak, "
                  f"got {expected_count} expected energies")
            # Fallback: take highest channel peak
            selected_peak = max(persistent_peaks,
                                key=lambda x: x['peak_index'])
            return [selected_peak['peak_index']]

    elif source_name.lower() == "cobalt":
        # For Cobalt: select 2 peaks with highest channels (1173, 1332 keV)
        if expected_count == 2:
            # Sort by channel position and take the 2 highest
            sorted_peaks = sorted(persistent_peaks,
                                  key=lambda x: x['peak_index'],
                                  reverse=True)
            selected_peaks = sorted_peaks[:2]
            # Return sorted by channel position for proper energy assignment
            selected_peaks.sort(key=lambda x: x['peak_index'])
            return [peak['peak_index'] for peak in selected_peaks]
        else:
            print(f"Warning: Cobalt expected to have 2 peaks, "
                  f"got {expected_count} expected energies")
            # Fallback: take highest channel peaks
            sorted_peaks = sorted(persistent_peaks,
                                  key=lambda x: x['peak_index'],
                                  reverse=True)
            selected_peaks = sorted_peaks[:expected_count]
            selected_peaks.sort(key=lambda x: x['peak_index'])
            return [peak['peak_index'] for peak in selected_peaks]
    
    else:
        # Default behavior for other sources: sort by energy (channel position)
        # and take the expected number of peaks
        sorted_peaks = sorted(persistent_peaks, key=lambda x: x['peak_index'])
        selected_peaks = sorted_peaks[:expected_count]
        return [peak['peak_index'] for peak in selected_peaks]


def calibrate_detector_array(sources_data: Dict[str, Dict],
                             expected_energies: Dict[str, List[float]]
                             ) -> Dict[int, Dict]:
    """
    Calibrate all detectors using data from multiple sources with proper
    two-step peak selection.

    Step 1: Select top-K peaks based on volumetric persistence
    Step 2: Apply source-specific selection rules to choose calibration peaks

    Parameters:
        sources_data (Dict[str, Dict]): Dictionary with source names as keys
                                       and their data as values
        expected_energies (Dict[str, List[float]]): Expected energies for
                                                   each source

    Returns:
        Dict[int, Dict]: Calibration results for each detector
    """
    # Determine number of detectors
    first_source = next(iter(sources_data.values()))
    n_detectors = first_source['spectra'].shape[1]
    
    calibration_results = {}
    
    for detector_idx in range(n_detectors):
        aggregated_channels = []
        aggregated_energies = []
        peaks_per_source = {}
        
        for source_name, source_data in sources_data.items():
            expected_energy_list = expected_energies[source_name]
            expected_count = len(expected_energy_list)
            
            # Get detected peaks for this detector (use calibration peaks)
            detector_peak_data = source_data['peaks'].get(detector_idx, {})
            detector_peaks = detector_peak_data.get('calibration_peaks', [])
            
            # Check if we have enough persistent peaks
            if len(detector_peaks) < expected_count:
                print(f"Warning: Detector {detector_idx}, "
                      f"Source {source_name}: "
                      f"Only {len(detector_peaks)} peaks detected "
                      f"(persistence-filtered), expected {expected_count} "
                      f"for calibration")
                continue

            # STEP 1: The peaks are already the top-K most persistent
            # These peaks come pre-filtered by the volumetric persistence
            # algorithm

            # STEP 2: Apply enhanced source-specific selection with validation
            selected_peak_channels = apply_source_specific_peak_selection_robust(
                detector_peaks, source_name, expected_energy_list
            )

            if len(selected_peak_channels) != expected_count:
                print(f"Warning: Detector {detector_idx}, "
                      f"Source {source_name}: "
                      f"Selected {len(selected_peak_channels)} peaks, "
                      f"expected {expected_count}")
                if len(selected_peak_channels) == 0:
                    continue

            # Map selected peaks to their corresponding energies
            # Sort both to ensure proper correspondence
            selected_peak_channels.sort()  # Sort by channel (energy)
            sorted_energies = sorted(expected_energy_list)  # Sort energies

            # Take only the number of peaks we actually found
            n_peaks_to_use = min(len(selected_peak_channels),
                                 len(sorted_energies))
            final_channels = selected_peak_channels[:n_peaks_to_use]
            final_energies = sorted_energies[:n_peaks_to_use]

            peaks_per_source[source_name] = final_channels
            aggregated_channels.extend(final_channels)
            aggregated_energies.extend(final_energies)

        # Perform calibration
        calibration = calibrate_detector(aggregated_channels,
                                         aggregated_energies)
        
        calibration_results[detector_idx] = {
            'calibration': calibration,
            'peaks_per_source': peaks_per_source,
            'calibration_data': {
                'channels': aggregated_channels,
                'energies': aggregated_energies
            }
        }
    
    return calibration_results


def find_poor_detectors(calibration_results: Dict[int, Dict], 
                        error_threshold: float = 0.05) -> List[int]:
    """
    Identify detectors with poor calibration performance.
    
    Parameters:
        calibration_results (Dict[int, Dict]): Calibration results
        error_threshold (float): CV error threshold for poor performance
        
    Returns:
        List[int]: List of detector indices with poor performance
    """
    poor_detectors = []
    
    for detector_idx, result in calibration_results.items():
        calib = result['calibration']
        cv_error = calib.get('cv_error')
        
        if cv_error is not None and cv_error > error_threshold:
            poor_detectors.append(detector_idx)
    
    return poor_detectors


def channel_to_energy(channels: Union[np.ndarray, List[float]], 
                      slope: float, 
                      intercept: float) -> np.ndarray:
    """
    Convert channel numbers to energy using calibration parameters.
    
    Parameters:
        channels (Union[np.ndarray, List[float]]): Channel numbers
        slope (float): Calibration slope
        intercept (float): Calibration intercept
        
    Returns:
        np.ndarray: Energy values
    """
    channels = np.array(channels)
    return slope * channels + intercept


def energy_to_channel(energies: Union[np.ndarray, List[float]], 
                      slope: float, 
                      intercept: float) -> np.ndarray:
    """
    Convert energy values to channel numbers using calibration parameters.
    
    Parameters:
        energies (Union[np.ndarray, List[float]]): Energy values
        slope (float): Calibration slope
        intercept (float): Calibration intercept
        
    Returns:
        np.ndarray: Channel numbers
    """
    energies = np.array(energies)
    return (energies - intercept) / slope


def create_parameter_grid(smoothing_config: Dict = None,
                          bins_factor_config: Dict = None,
                          threshold_config: Dict = None,
                          width_config: Dict = None,
                          prominence_config: Dict = None,
                          distance_config: Dict = None,
                          use_defaults: bool = True) -> Dict:
    """
    Create a parameter grid for volumetric persistence peak detection.
    
    Parameters:
        smoothing_config (Dict): {'min': float, 'max': float, 'steps': int}
        bins_factor_config (Dict): {'min': int, 'max': int, 'steps': int}
        threshold_config (Dict): {'min': float, 'max': float, 'steps': int}
        width_config (Dict): {'min': float, 'max': float, 'steps': int}
        prominence_config (Dict): {'min': float, 'max': float, 'steps': int}
        distance_config (Dict): {'min': float, 'max': float, 'steps': int}
        use_defaults (bool): If True, use default values for missing configs
        
    Returns:
        Dict: Parameter dictionary compatible with detection functions
        
    Example:
        >>> params = create_parameter_grid(
        ...     smoothing_config={'min': 1, 'max': 5, 'steps': 3},
        ...     threshold_config={'min': 0.01, 'max': 0.1, 'steps': 4}
        ... )
        >>> # Results in smoothing_range=[1, 3, 5] and threshold_range=[0.01, 0.04, 0.07, 0.1]
    """
    # Default configurations
    defaults = {
        'smoothing': {'min': 1, 'max': 5, 'steps': 3},
        'bins_factor': {'min': 1, 'max': 2, 'steps': 2},
        'threshold': {'min': 0.01, 'max': 0.1, 'steps': 3},
        'width': {'min': 1, 'max': 5, 'steps': 3},
        'prominence': {'min': 0.1, 'max': 0.5, 'steps': 3},
        'distance': {'min': 5, 'max': 15, 'steps': 3}
    }
    
    # Use provided configs or defaults
    configs = {
        'smoothing': smoothing_config or (defaults['smoothing'] if use_defaults else None),
        'bins_factor': bins_factor_config or (defaults['bins_factor'] if use_defaults else None),
        'threshold': threshold_config or (defaults['threshold'] if use_defaults else None),
        'width': width_config or (defaults['width'] if use_defaults else None),
        'prominence': prominence_config or (defaults['prominence'] if use_defaults else None),
        'distance': distance_config or (defaults['distance'] if use_defaults else None)
    }
    
    def _create_range(config, param_name):
        """Create a range based on configuration."""
        if config is None:
            raise ValueError(f"Configuration for {param_name} is required when use_defaults=False")
        
        min_val = config['min']
        max_val = config['max']
        steps = config['steps']
        
        if steps == 1:
            if param_name in ['bins_factor', 'smoothing', 'distance']:
                return [int(min_val)]
            else:
                return [min_val]
        elif steps == 2:
            if param_name in ['bins_factor', 'smoothing', 'distance']:
                return [int(min_val), int(max_val)]
            else:
                return [min_val, max_val]
        else:
            # Parameters that should be integers
            if param_name in ['bins_factor', 'smoothing', 'distance']:
                # For integer parameters, use integer values and convert to regular Python int
                values = np.linspace(min_val, max_val, steps, dtype=int)
                return [int(v) for v in values]
            elif param_name == 'width':
                # Width can be float, but round to reasonable precision
                values = np.linspace(min_val, max_val, steps)
                return [round(float(v), 1) for v in values]
            else:
                return list(np.linspace(min_val, max_val, steps))
    
    # Create parameter ranges
    parameter_grid = {}
    
    for param_name, config in configs.items():
        if config is not None:
            range_name = f"{param_name}_range"
            parameter_grid[range_name] = _create_range(config, param_name)
    
    # Add additional parameters with defaults
    parameter_grid['merging_range'] = 5
    parameter_grid['tol'] = 1
    parameter_grid['parallel'] = True
    
    return parameter_grid


def print_parameter_grid(parameter_grid: Dict):
    """
    Print a formatted view of the parameter grid.
    
    Parameters:
        parameter_grid (Dict): Parameter grid dictionary
    """
    print("📊 Volumetric Persistence Parameter Grid")
    print("=" * 50)
    
    param_info = [
        ('smoothing_range', 'Smoothing window sizes'),
        ('bins_factor_range', 'Channel binning factors'),
        ('threshold_range', 'Peak detection thresholds'),
        ('width_range', 'Expected peak widths'),
        ('prominence_range', 'Peak prominence values'),
        ('distance_range', 'Minimum peak distances')
    ]
    
    for param_key, description in param_info:
        if param_key in parameter_grid:
            values = parameter_grid[param_key]
            if len(values) <= 5:
                values_str = str(values)
            else:
                values_str = f"[{values[0]}, {values[1]}, ..., {values[-2]}, {values[-1]}] ({len(values)} values)"
            print(f"{description:.<35} {values_str}")
    
    # Calculate total combinations
    total_combinations = 1
    for param_key, _ in param_info:
        if param_key in parameter_grid:
            total_combinations *= len(parameter_grid[param_key])
    
    print(f"\n🔢 Total parameter combinations: {total_combinations:,}")
    
    # Performance warning
    if total_combinations > 1000:
        print("⚠️  Warning: Large parameter grid may be slow. Consider reducing ranges.")
    elif total_combinations > 100:
        print("💡 Tip: Consider enabling parallel processing for faster execution.")


def get_optimized_parameter_suggestions(energy_category: str = "medium_energy") -> Dict:
    """
    Get suggested parameter configurations for different energy categories.
    
    Parameters:
        energy_category (str): 'low_energy', 'medium_energy', 'high_energy', 'multi_peak'
        
    Returns:
        Dict: Suggested parameter configurations
    """
    suggestions = {
        'low_energy': {
            'smoothing': {'min': 1, 'max': 3, 'steps': 2},
            'bins_factor': {'min': 1, 'max': 1, 'steps': 1},
            'threshold': {'min': 0.01, 'max': 0.05, 'steps': 3},
            'width': {'min': 1, 'max': 3, 'steps': 3},
            'prominence': {'min': 0.05, 'max': 0.2, 'steps': 3},
            'distance': {'min': 3, 'max': 8, 'steps': 3}
        },
        'medium_energy': {
            'smoothing': {'min': 1, 'max': 5, 'steps': 3},
            'bins_factor': {'min': 1, 'max': 2, 'steps': 2},
            'threshold': {'min': 0.01, 'max': 0.1, 'steps': 3},
            'width': {'min': 2, 'max': 6, 'steps': 3},
            'prominence': {'min': 0.1, 'max': 0.5, 'steps': 3},
            'distance': {'min': 5, 'max': 15, 'steps': 3}
        },
        'high_energy': {
            'smoothing': {'min': 3, 'max': 8, 'steps': 3},
            'bins_factor': {'min': 1, 'max': 3, 'steps': 3},
            'threshold': {'min': 0.05, 'max': 0.2, 'steps': 3},
            'width': {'min': 3, 'max': 8, 'steps': 3},
            'prominence': {'min': 0.2, 'max': 0.6, 'steps': 3},
            'distance': {'min': 8, 'max': 20, 'steps': 3}
        },
        'multi_peak': {
            'smoothing': {'min': 1, 'max': 5, 'steps': 3},
            'bins_factor': {'min': 1, 'max': 2, 'steps': 2},
            'threshold': {'min': 0.01, 'max': 0.05, 'steps': 3},
            'width': {'min': 1, 'max': 8, 'steps': 4},
            'prominence': {'min': 0.05, 'max': 0.3, 'steps': 4},
            'distance': {'min': 5, 'max': 20, 'steps': 4}
        }
    }
    
    if energy_category not in suggestions:
        print(f"Warning: Unknown energy category '{energy_category}'. Using 'medium_energy'.")
        energy_category = 'medium_energy'
    
    return suggestions[energy_category]


def create_custom_detection_params(source_name: str = None,
                                   energy_category: str = None,
                                   custom_grid: Dict = None,
                                   **kwargs) -> Dict:
    """
    Create detection parameters with custom hyperparameter ranges.
    
    Parameters:
        source_name (str): Name of radioactive source (auto-detects energy category)
        energy_category (str): Energy category for optimization suggestions
        custom_grid (Dict): Complete custom parameter grid
        **kwargs: Individual parameter configurations (e.g., smoothing_config={...})
        
    Returns:
        Dict: Detection parameters ready for use
        
    Example:
        >>> # Use source-based optimization
        >>> params = create_custom_detection_params(source_name="Cesium")
        
        >>> # Use energy category optimization  
        >>> params = create_custom_detection_params(energy_category="high_energy")
        
        >>> # Custom single parameter
        >>> params = create_custom_detection_params(
        ...     smoothing_config={'min': 1, 'max': 10, 'steps': 5}
        ... )
        
        >>> # Complete custom grid
        >>> params = create_custom_detection_params(
        ...     custom_grid={
        ...         'smoothing_range': [1, 3, 5],
        ...         'threshold_range': [0.01, 0.05, 0.1]
        ...     }
        ... )
    """
    if custom_grid is not None:
        # Use provided custom grid directly
        params = custom_grid.copy()
        # Ensure essential parameters are present
        params.setdefault('merging_range', 5)
        params.setdefault('tol', 1)
        params.setdefault('parallel', True)
        return params
    
    # Determine energy category
    if source_name and not energy_category:
        from source_config import get_source_config
        try:
            source_config = get_source_config(source_name)
            energy_category = source_config['energy_category']
        except:
            energy_category = 'medium_energy'
    
    if not energy_category:
        energy_category = 'medium_energy'
    
    # Get base suggestions
    suggestions = get_optimized_parameter_suggestions(energy_category)
    
    # Override with any custom configurations
    param_configs = {}
    for param in ['smoothing', 'bins_factor', 'threshold', 'width', 'prominence', 'distance']:
        config_key = f"{param}_config"
        if config_key in kwargs:
            param_configs[param] = kwargs[config_key]
        else:
            param_configs[param] = suggestions[param]
    
    # Create parameter grid
    return create_parameter_grid(
        smoothing_config=param_configs['smoothing'],
        bins_factor_config=param_configs['bins_factor'],
        threshold_config=param_configs['threshold'],
        width_config=param_configs['width'],
        prominence_config=param_configs['prominence'],
        distance_config=param_configs['distance']
    )
