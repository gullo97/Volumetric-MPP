"""
Test script to demonstrate the improved plot_combined_source_spectra function
that shows ALL detected peaks in the barcode plot instead of just the top-k.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'calibration'))

from calibration.calibration_plots import plot_combined_source_spectra
from calibration.source_config import get_detection_params
import numpy as np

def create_test_data():
    """Create synthetic test data to demonstrate the function."""
    # Create synthetic spectrum with multiple peaks
    n_channels = 2048
    n_detectors = 1
    detector_idx = 0
    
    # Create test spectra for multiple sources
    sources_data = {}
    
    # Cesium source (simulate 662 keV peak)
    cesium_spectrum = np.random.poisson(50, (n_channels, n_detectors))  # Background
    # Add peak at channel ~1300 (simulating 662 keV)
    peak_channels = [1300]
    for peak_ch in peak_channels:
        for ch in range(max(0, peak_ch-10), min(n_channels, peak_ch+10)):
            height = 1000 * np.exp(-0.5 * ((ch - peak_ch) / 5) ** 2)
            cesium_spectrum[ch, 0] += int(height)
    
    # Add some smaller peaks to show difference between top-k and all peaks
    small_peaks = [800, 900, 1000, 1100, 1200, 1400, 1500, 1600]
    for peak_ch in small_peaks:
        for ch in range(max(0, peak_ch-5), min(n_channels, peak_ch+5)):
            height = 200 * np.exp(-0.5 * ((ch - peak_ch) / 3) ** 2)
            cesium_spectrum[ch, 0] += int(height)
    
    # Simulate detected peaks (top-k would typically be 2-3, but there are more small peaks)
    cesium_peaks = {
        detector_idx: [
            {'peak_index': 1300, 'persistence': 0.8},  # Main peak
            {'peak_index': 800, 'persistence': 0.2},   # Top-k might stop here
            {'peak_index': 1200, 'persistence': 0.15}  # But there are more peaks
        ]
    }
    
    sources_data['Cesium'] = {
        'spectra': cesium_spectrum,
        'peaks': cesium_peaks
    }
    
    # Cobalt source (simulate 1173 and 1332 keV peaks)
    cobalt_spectrum = np.random.poisson(40, (n_channels, n_detectors))
    cobalt_peaks_ch = [1700, 1900]  # Simulating higher energy peaks
    for peak_ch in cobalt_peaks_ch:
        for ch in range(max(0, peak_ch-8), min(n_channels, peak_ch+8)):
            height = 800 * np.exp(-0.5 * ((ch - peak_ch) / 4) ** 2)
            cobalt_spectrum[ch, 0] += int(height)
    
    # Add more small peaks
    more_small_peaks = [500, 600, 700, 1100, 1400, 1600, 1800]
    for peak_ch in more_small_peaks:
        for ch in range(max(0, peak_ch-4), min(n_channels, peak_ch+4)):
            height = 150 * np.exp(-0.5 * ((ch - peak_ch) / 2) ** 2)
            cobalt_spectrum[ch, 0] += int(height)
    
    cobalt_peaks = {
        detector_idx: [
            {'peak_index': 1700, 'persistence': 0.7},
            {'peak_index': 1900, 'persistence': 0.65},
            {'peak_index': 1400, 'persistence': 0.18}  # Top-k might include only first 2-3
        ]
    }
    
    sources_data['Cobalt'] = {
        'spectra': cobalt_spectrum,
        'peaks': cobalt_peaks
    }
    
    return sources_data

def main():
    """Test the improved plotting function."""
    print("Testing improved plot_combined_source_spectra function...")
    
    # Create test data
    sources_data = create_test_data()
    detector_idx = 0
    
    # Get detection parameters
    detection_params = get_detection_params("medium_energy")
    
    print("\\n1. Testing with show_all_peaks_in_barcode=False (original behavior):")
    print("   This will show only the pre-selected top-k peaks in the barcode")
    
    plot_combined_source_spectra(
        detector_idx=detector_idx,
        sources_data=sources_data,
        show_all_peaks_in_barcode=False
    )
    
    print("\\n2. Testing with show_all_peaks_in_barcode=True (improved behavior):")
    print("   This will re-detect and show ALL peaks in the barcode")
    
    plot_combined_source_spectra(
        detector_idx=detector_idx,
        sources_data=sources_data,
        show_all_peaks_in_barcode=True,
        detection_params=detection_params
    )
    
    print("\\nThe difference:")
    print("- Top panel (spectrum): Shows the same peaks in both cases")
    print("- Bottom panel (barcode): ")
    print("  * False: Shows only the pre-selected top-k peaks")
    print("  * True: Shows ALL detected peaks, including smaller ones")
    print("\\nThis gives you a complete view of all peak candidates!")

if __name__ == "__main__":
    main()
