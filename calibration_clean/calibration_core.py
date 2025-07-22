"""
Core calibration functions for detector calibration using volumetric persistence.

This module provides simplified, working functions for detector calibration
without the complex parameter optimization that was causing issues.
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress
from typing import Dict, List, Tuple, Union, Optional
import sys
import os

# Add parent directory to path to import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import find_peaks_volumetric_persistence
from source_config import GLOBAL_DETECTION_PARAMS


def load_data_from_excel(file_path: str, sheet_name: str) -> np.ndarray:
    """Load spectrum data from Excel sheet."""
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    data_numeric = df.apply(pd.to_numeric, errors='coerce').dropna(how='all')
    if data_numeric.empty:
        raise ValueError(f"No numeric data found in sheet {sheet_name}.")
    return data_numeric.values


def load_data_from_csv(file_path: str) -> np.ndarray:
    """Load spectrum data from CSV file."""
    df = pd.read_csv(file_path)
    data_numeric = df.apply(pd.to_numeric, errors='coerce').dropna(how='all')
    if data_numeric.empty:
        raise ValueError(f"No numeric data found in CSV file {file_path}.")
    return data_numeric.values


def detect_peaks_single_detector(spectrum: np.ndarray, 
                                detection_params: dict,
                                top_k: int = 5) -> List[int]:
    """
    Detect peaks for a single detector using volumetric persistence.
    
    Parameters:
        spectrum: 1D spectrum data
        detection_params: Detection parameters
        top_k: Number of top peaks to return
        
    Returns:
        List of peak channel positions
    """
    try:
        # Use the global detection parameters
        peak_results = find_peaks_volumetric_persistence(
            spectrum, 
            **detection_params
        )
        
        # Extract peak indices from the results (which are dictionaries)
        peak_indices = [peak['peak_index'] for peak in peak_results]
        
        # Return top_k peaks sorted by channel position
        if len(peak_indices) > top_k:
            peak_indices = peak_indices[:top_k]
            
        return sorted(peak_indices)
        
    except Exception as e:
        print(f"Warning: Peak detection failed: {e}")
        return []


def apply_channel_range_filter(peaks: List[int], 
                             channel_ranges: List[Tuple[int, int]]) -> List[int]:
    """
    Filter peaks by expected channel ranges.
    
    Parameters:
        peaks: List of peak channel positions
        channel_ranges: List of (min, max) channel ranges
        
    Returns:
        Filtered peaks within expected ranges
    """
    if not channel_ranges:
        return peaks
        
    filtered_peaks = []
    for peak in peaks:
        for min_ch, max_ch in channel_ranges:
            if min_ch <= peak <= max_ch:
                filtered_peaks.append(peak)
                break
    
    return filtered_peaks


def select_physics_based_peaks(peaks: List[int], 
                             source_name: str,
                             expected_count: int) -> List[int]:
    """
    Apply physics-based peak selection rules.
    
    Parameters:
        peaks: Candidate peaks
        source_name: Name of the radioactive source
        expected_count: Expected number of peaks
        
    Returns:
        Selected peaks based on physics rules
    """
    if not peaks:
        return []
    
    # Sort peaks by channel (energy)
    sorted_peaks = sorted(peaks)
    
    if source_name in ["Sodium", "Cesium", "Americium", "Manganese"]:
        # Single peak sources - select highest energy peak
        return [sorted_peaks[-1]] if sorted_peaks else []
    
    elif source_name == "Cobalt":
        # Two peak source - select two highest energy peaks
        return sorted_peaks[-2:] if len(sorted_peaks) >= 2 else sorted_peaks
    
    else:
        # Multi-peak sources - select from lowest to highest energy
        return sorted_peaks[:expected_count] if len(sorted_peaks) >= expected_count else sorted_peaks


def process_source_data(file_path: str, 
                       source_config: dict,
                       detection_params: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
    """
    Process spectral data for a single radioactive source.
    
    Parameters:
        file_path: Path to data file
        source_config: Source configuration including sheet_name/file_path and top_k
        detection_params: Detection parameters (uses global if None)
        
    Returns:
        Tuple of (spectra_array, peaks_dict)
    """
    # Use global parameters if none provided
    if detection_params is None:
        detection_params = GLOBAL_DETECTION_PARAMS.copy()
    
    # Load data
    if 'sheet_name' in source_config:
        # Excel file
        spectra = load_data_from_excel(file_path, source_config['sheet_name'])
    else:
        # CSV file
        spectra = load_data_from_csv(source_config['file_path'])
    
    n_channels, n_detectors = spectra.shape
    top_k = source_config.get('top_k', 5)
    
    # Detect peaks for each detector
    peaks_data = {}
    
    for detector_idx in range(n_detectors):
        spectrum = spectra[:, detector_idx]
        
        # Detect peaks using volumetric persistence
        detected_peaks = detect_peaks_single_detector(
            spectrum, detection_params, top_k
        )
        
        # Apply channel range filtering if available
        if 'channel_ranges' in source_config:
            detected_peaks = apply_channel_range_filter(
                detected_peaks, source_config['channel_ranges']
            )
        
        # Apply physics-based selection
        if 'source_name' in source_config:
            expected_count = len(source_config.get('energies', []))
            selected_peaks = select_physics_based_peaks(
                detected_peaks, source_config['source_name'], expected_count
            )
        else:
            selected_peaks = detected_peaks
        
        peaks_data[detector_idx] = {
            'calibration_peaks': selected_peaks,
            'all_detected_peaks': detected_peaks  # For visualization
        }
    
    return spectra, peaks_data


def calibrate_single_detector(peaks_per_source: Dict[str, List[int]], 
                            expected_energies: Dict[str, List[float]]) -> Dict:
    """
    Perform linear calibration for a single detector.
    
    Parameters:
        peaks_per_source: Dictionary mapping source names to peak channels
        expected_energies: Dictionary mapping source names to expected energies
        
    Returns:
        Calibration parameters and statistics
    """
    # Collect channel-energy pairs
    channels = []
    energies = []
    
    for source_name, peak_channels in peaks_per_source.items():
        if source_name in expected_energies:
            source_energies = expected_energies[source_name]
            
            # Match peaks to energies (assume sorted order)
            n_matches = min(len(peak_channels), len(source_energies))
            
            for i in range(n_matches):
                channels.append(peak_channels[i])
                energies.append(source_energies[i])
    
    # Require at least 2 points for linear calibration
    if len(channels) < 2:
        return {
            'slope': None,
            'intercept': None,
            'r_squared': None,
            'cv_error': None,
            'p_value': None,
            'std_err': None,
            'n_points': len(channels)
        }
    
    # Perform linear regression
    channels = np.array(channels)
    energies = np.array(energies)
    
    slope, intercept, r_value, p_value, std_err = linregress(channels, energies)
    
    # Calculate cross-validation error (leave-one-out)
    cv_errors = []
    n_points = len(channels)
    
    for i in range(n_points):
        # Leave out point i
        train_channels = np.delete(channels, i)
        train_energies = np.delete(energies, i)
        
        if len(train_channels) >= 2:
            # Fit on remaining points
            temp_slope, temp_intercept, _, _, _ = linregress(train_channels, train_energies)
            
            # Predict left-out point
            predicted_energy = temp_slope * channels[i] + temp_intercept
            cv_errors.append(abs(predicted_energy - energies[i]))
    
    cv_error = np.mean(cv_errors) if cv_errors else None
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'cv_error': cv_error,
        'p_value': p_value,
        'std_err': std_err,
        'n_points': n_points
    }


def calibrate_detector_array(sources_data: Dict[str, Dict], 
                           expected_energies: Dict[str, List[float]]) -> Dict[int, Dict]:
    """
    Calibrate all detectors using data from multiple sources.
    
    Parameters:
        sources_data: Dictionary containing spectra and peaks for each source
        expected_energies: Expected energies for each source
        
    Returns:
        Dictionary mapping detector indices to calibration results
    """
    # Get detector count from first source
    first_source = next(iter(sources_data.values()))
    n_detectors = first_source['spectra'].shape[1]
    
    calibration_results = {}
    
    for detector_idx in range(n_detectors):
        # Collect peaks for this detector from all sources
        peaks_per_source = {}
        
        for source_name, source_data in sources_data.items():
            detector_peaks = source_data['peaks'].get(detector_idx, {})
            
            # Get calibration peaks
            if isinstance(detector_peaks, dict) and 'calibration_peaks' in detector_peaks:
                calibration_peaks = detector_peaks['calibration_peaks']
            else:
                # Fallback for old structure
                calibration_peaks = detector_peaks if isinstance(detector_peaks, list) else []
            
            if calibration_peaks:
                peaks_per_source[source_name] = calibration_peaks
        
        # Perform calibration
        calibration = calibrate_single_detector(peaks_per_source, expected_energies)
        
        calibration_results[detector_idx] = {
            'calibration': calibration,
            'peaks_per_source': peaks_per_source
        }
    
    return calibration_results


def find_poor_detectors(calibration_results: Dict[int, Dict], 
                       cv_threshold: float = 0.1) -> List[int]:
    """
    Identify detectors with poor calibration quality.
    
    Parameters:
        calibration_results: Calibration results for all detectors
        cv_threshold: CV error threshold in MeV
        
    Returns:
        List of detector indices with poor calibration
    """
    poor_detectors = []
    
    for detector_idx, result in calibration_results.items():
        calib = result['calibration']
        cv_error = calib.get('cv_error')
        
        if cv_error is None or cv_error > cv_threshold:
            poor_detectors.append(detector_idx)
    
    return poor_detectors


def channel_to_energy(channels: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Convert channel numbers to energy using calibration parameters."""
    return slope * channels + intercept


def energy_to_channel(energies: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Convert energies to channel numbers using calibration parameters."""
    return (energies - intercept) / slope
