"""
Calibration Visualization Functions for Volumetric MPP

This module provides plotting and visualization functions for detector calibration
results and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Optional
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration_core import channel_to_energy, detect_peaks_single_detector

# Font configuration for plots
FONTSIZE = {
    "label": 18,
    "title": 18,
    "ticks": 16,
    "legend": 16
}

# Set Times New Roman as default font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'


def plot_calibration_quality_overview(calibration_results: Dict[int, Dict],
                                       save_path: Optional[str] = None):
    """
    Create overview plots of calibration quality across all detectors.
    
    Parameters:
        calibration_results (Dict[int, Dict]): Calibration results
        save_path (Optional[str]): Path to save plots
    """
    # Extract data for plotting
    detectors, slopes, intercepts, cv_errors = [], [], [], []
    
    for det, result in calibration_results.items():
        calib = result.get('calibration', {})
        if calib.get('slope') is not None:
            detectors.append(det)
            slopes.append(calib['slope'])
            intercepts.append(calib['intercept'])
            cv_errors.append(calib['cv_error'])
    
    detectors = np.array(detectors)
    slopes = np.array(slopes)
    intercepts = np.array(intercepts)
    cv_errors = np.array(cv_errors)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Histogram of CV errors
    axes[0, 0].hist(cv_errors, bins=20, color='skyblue', 
                    edgecolor='k', alpha=0.85)
    axes[0, 0].set_title('CV Error Distribution', fontsize=FONTSIZE["title"])
    axes[0, 0].set_xlabel('CV Error [MeV]', fontsize=FONTSIZE["label"])
    axes[0, 0].set_ylabel('Frequency', fontsize=FONTSIZE["label"])
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    axes[0, 0].tick_params(axis='both', labelsize=FONTSIZE["ticks"])
    
    # 2. Boxplot of CV errors
    box_plot = axes[0, 1].boxplot(cv_errors, vert=True, patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightgreen')
    box_plot['medians'][0].set_color('red')
    axes[0, 1].set_title('CV Error Boxplot', fontsize=FONTSIZE["title"])
    axes[0, 1].set_ylabel('CV Error [MeV]', fontsize=FONTSIZE["label"])
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    axes[0, 1].tick_params(axis='both', labelsize=FONTSIZE["ticks"])
    
    # 3. Detector vs CV Error scatter
    sc1 = axes[0, 2].scatter(detectors, cv_errors, c=cv_errors, 
                             cmap='coolwarm', s=60, edgecolor='k')
    axes[0, 2].set_title('CV Error by Detector', fontsize=FONTSIZE["title"])
    axes[0, 2].set_xlabel('Detector Index', fontsize=FONTSIZE["label"])
    axes[0, 2].set_ylabel('CV Error [MeV]', fontsize=FONTSIZE["label"])
    cbar1 = plt.colorbar(sc1, ax=axes[0, 2])
    cbar1.set_label("CV Error [MeV]", fontsize=FONTSIZE["label"])
    axes[0, 2].grid(True, linestyle='--', alpha=0.6)
    axes[0, 2].tick_params(axis='both', labelsize=FONTSIZE["ticks"])
    
    # 4. Slope vs Intercept scatter
    sc2 = axes[1, 0].scatter(slopes, intercepts, c=cv_errors, 
                             cmap='coolwarm', s=60, edgecolor='k')
    axes[1, 0].set_title('Slope vs Intercept', fontsize=FONTSIZE["title"])
    axes[1, 0].set_xlabel('Slope', fontsize=FONTSIZE["label"])
    axes[1, 0].set_ylabel('Intercept', fontsize=FONTSIZE["label"])
    cbar2 = plt.colorbar(sc2, ax=axes[1, 0])
    cbar2.set_label("CV Error [MeV]", fontsize=FONTSIZE["label"])
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)
    axes[1, 0].tick_params(axis='both', labelsize=FONTSIZE["ticks"])
    
    # 5. Calibration parameters vs detector
    axes[1, 1].plot(detectors, slopes, 'o-', label="Slope", 
                    markersize=8, linewidth=2)
    axes[1, 1].plot(detectors, intercepts, 's-', label="Intercept", 
                    markersize=8, linewidth=2)
    axes[1, 1].set_title('Calibration Parameters', fontsize=FONTSIZE["title"])
    axes[1, 1].set_xlabel('Detector Index', fontsize=FONTSIZE["label"])
    axes[1, 1].set_ylabel('Parameter Value', fontsize=FONTSIZE["label"])
    axes[1, 1].legend(fontsize=FONTSIZE["legend"])
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)
    axes[1, 1].tick_params(axis='both', labelsize=FONTSIZE["ticks"])
    
    # 6. R-squared distribution
    r_squared_values = [result['calibration']['r_squared'] 
                        for result in calibration_results.values()
                        if result['calibration']['r_squared'] is not None]
    axes[1, 2].hist(r_squared_values, bins=20, color='lightcoral', 
                    edgecolor='k', alpha=0.85)
    axes[1, 2].set_title('R² Distribution', fontsize=FONTSIZE["title"])
    axes[1, 2].set_xlabel('R²', fontsize=FONTSIZE["label"])
    axes[1, 2].set_ylabel('Frequency', fontsize=FONTSIZE["label"])
    axes[1, 2].grid(True, linestyle='--', alpha=0.6)
    axes[1, 2].tick_params(axis='both', labelsize=FONTSIZE["ticks"])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_single_detector_calibration(detector_idx: int, 
                                      calibration_result: Dict,
                                      save_path: Optional[str] = None):
    """
    Plot calibration curve for a single detector with LOOCV error bars.
    
    Parameters:
        detector_idx (int): Detector index
        calibration_result (Dict): Calibration result for this detector
        save_path (Optional[str]): Path to save plot
    """
    calib_data = calibration_result['calibration_data']
    calibration = calibration_result['calibration']
    
    channels = np.array(calib_data['channels'])
    energies = np.array(calib_data['energies'])
    
    # Compute LOOCV predictions and errors
    from scipy.stats import linregress
    
    loocv_errors = []
    n = len(channels)
    for i in range(n):
        x_train = np.delete(channels, i)
        y_train = np.delete(energies, i)
        if len(x_train) < 2:
            loocv_errors.append(0)
            continue
        s, b, _, _, _ = linregress(x_train, y_train)
        pred = s * channels[i] + b
        loocv_errors.append(abs(pred - energies[i]))
    
    loocv_errors = np.array(loocv_errors)
    
    # Create calibration line
    slope = calibration['slope']
    intercept = calibration['intercept']
    x_range = np.linspace(0, 2047, 100)
    y_range = slope * x_range + intercept
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(channels, energies, yerr=loocv_errors, fmt='o', 
                 capsize=5, label='Calibration points', markersize=6,
                 markerfacecolor='white', markeredgecolor='black')
    
    line_label = f'Fit: E = {slope:.3f}×Ch + {intercept:.3f}'
    plt.plot(x_range, y_range, label=line_label, color='red', 
             linestyle='-', linewidth=2)
    
    plt.xlabel('Channel', fontsize=FONTSIZE["label"])
    plt.ylabel('Energy [MeV]', fontsize=FONTSIZE["label"])
    plt.title(f'Detector {detector_idx} Calibration', 
              fontsize=FONTSIZE["title"])
    plt.legend(fontsize=FONTSIZE["legend"])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tick_params(axis='both', labelsize=FONTSIZE["ticks"])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Detector {detector_idx} average LOOCV error: "
          f"{np.mean(loocv_errors):.4f} MeV")


def plot_combined_source_spectra(detector_idx: int,
                                  sources_data: Dict[str, Dict],
                                  source_colors: Optional[Dict[str, str]] = None,
                                  save_path: Optional[str] = None,
                                  show_all_peaks_in_barcode: bool = True,
                                  detection_params: Optional[Dict] = None):
    """
    Plot combined spectra from multiple sources for a given detector.
    
    Parameters:
        detector_idx (int): Detector index to plot
        sources_data (Dict[str, Dict]): Data from all sources
        source_colors (Optional[Dict[str, str]]): Colors for each source
        save_path (Optional[str]): Path to save plot
        show_all_peaks_in_barcode (bool): If True, show ALL detected peaks
                                         in barcode, not just the top-k
        detection_params (Optional[Dict]): Detection parameters needed if
                                          show_all_peaks_in_barcode is True
    """
    if source_colors is None:
        source_colors = {
            "Cesium": "#CBAC88",
            "Sodium": "#69995D",
            "Cobalt": "#C94040"
        }
    
    fig, (ax_spec, ax_barcode) = plt.subplots(2, 1, figsize=(10, 8),
                                              sharex=True)
    
    max_channel = 0
    all_peaks_for_barcode = {}  # Store all peaks for barcode plot
    
    for source_name, source_data in sources_data.items():
        spectra = source_data['spectra']
        peaks = source_data['peaks']
        
        if detector_idx >= spectra.shape[1]:
            continue
        
        spectrum = spectra[:, detector_idx]
        channels = np.arange(len(spectrum))
        max_channel = max(max_channel, len(spectrum))
        
        # Normalize spectrum
        norm_spec = (spectrum / np.max(spectrum)
                     if np.max(spectrum) > 0 else spectrum)
        
        color = source_colors.get(source_name, 'black')
        
        # Plot spectrum
        ax_spec.plot(channels, norm_spec, color=color, linewidth=1.5,
                     label=f"{source_name}")
        
        # Plot detected peaks (only the selected calibration peaks)
        if detector_idx in peaks:
            peak_data = peaks[detector_idx]
            if isinstance(peak_data, dict) and 'calibration_peaks' in peak_data:
                # New structure: extract calibration peaks
                peak_channels = [peak['peak_index']
                               for peak in peak_data['calibration_peaks']]
            else:
                # Old structure: direct list of peaks
                peak_channels = [peak['peak_index'] for peak in peak_data]
                
            peak_heights = [norm_spec[int(ch)] for ch in peak_channels
                           if int(ch) < len(norm_spec)]
            peak_channels = [ch for ch in peak_channels
                           if int(ch) < len(norm_spec)]
            
            ax_spec.scatter(peak_channels, peak_heights, color=color,
                           s=80, marker='o', edgecolor='black',
                           zorder=5, alpha=0.8)
        
        # For barcode plot: get all peaks for visualization
        if show_all_peaks_in_barcode:
            # Use the stored all_peaks if available
            if detector_idx in peaks:
                peak_data = peaks[detector_idx]
                if isinstance(peak_data, dict) and 'all_peaks' in peak_data:
                    all_peaks_for_barcode[source_name] = peak_data['all_peaks']
                elif detection_params is not None:
                    # Fallback: re-detect with extended range
                    all_peaks = detect_peaks_single_detector(
                        spectrum, detection_params, top_k=1000
                    )
                    # Handle new return format (calibration_peaks, all_peaks)
                    if isinstance(all_peaks, tuple):
                        all_peaks_for_barcode[source_name] = all_peaks[1]
                    else:
                        all_peaks_for_barcode[source_name] = all_peaks
                else:
                    # Use existing peaks as fallback
                    if isinstance(peak_data, dict) and 'calibration_peaks' in peak_data:
                        all_peaks_for_barcode[source_name] = peak_data['calibration_peaks']
                    else:
                        all_peaks_for_barcode[source_name] = peak_data
        else:
            # Use only the calibration peaks
            if detector_idx in peaks:
                peak_data = peaks[detector_idx]
                if isinstance(peak_data, dict) and 'calibration_peaks' in peak_data:
                    all_peaks_for_barcode[source_name] = peak_data['calibration_peaks']
                else:
                    all_peaks_for_barcode[source_name] = peak_data
    
    # Barcode plot with all peaks
    for source_name, all_peaks in all_peaks_for_barcode.items():
        color = source_colors.get(source_name, 'black')
        for peak in all_peaks:
            persistence = peak['persistence']
            channel = peak['peak_index']
            ax_barcode.vlines(channel, 0, persistence, colors=color,
                              lw=2, alpha=0.95)
            ax_barcode.scatter(channel, persistence, color=color,
                               s=50, edgecolor='k', zorder=3)
    
    # Format plots
    ax_spec.set_ylabel('Normalized Counts', fontsize=FONTSIZE["label"])
    ax_spec.set_title(f'Combined Spectra - Detector {detector_idx}', 
                     fontsize=FONTSIZE["title"])
    ax_spec.legend(fontsize=FONTSIZE["legend"])
    ax_spec.grid(True, linestyle='--', alpha=0.6)
    ax_spec.tick_params(axis='both', labelsize=FONTSIZE["ticks"])
    
    ax_barcode.set_xlabel('Channel', fontsize=FONTSIZE["label"])
    ax_barcode.set_ylabel('Peak Persistence', fontsize=FONTSIZE["label"])
    
    # Update title based on what peaks are shown
    if show_all_peaks_in_barcode and detection_params is not None:
        barcode_title = 'All Detected Peaks - Persistence Barcode'
    else:
        barcode_title = 'Selected Peaks - Persistence Barcode'
    ax_barcode.set_title(barcode_title, fontsize=FONTSIZE["title"])
    
    ax_barcode.grid(True, linestyle='--', alpha=0.6)
    ax_barcode.tick_params(axis='both', labelsize=FONTSIZE["ticks"])
    
    if max_channel > 0:
        ax_spec.set_xlim(0, max_channel)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_3d_energy_spectrum(sources_data: Dict[str, Dict],
                            calibration_results: Dict[int, Dict],
                            source_name: str,
                            poor_detectors: List[int],
                            count_threshold: int = 100,
                            save_path: Optional[str] = None):
    """
    Create 3D spectrum plot in energy space.
    
    Parameters:
        sources_data (Dict[str, Dict]): Data from all sources
        calibration_results (Dict[int, Dict]): Calibration results
        source_name (str): Source to visualize
        poor_detectors (List[int]): Detectors to exclude
        count_threshold (int): Minimum count threshold for filtering
        save_path (Optional[str]): Path to save plot
    """
    if source_name not in sources_data:
        print(f"Source {source_name} not found in data")
        return
    
    spectrum = sources_data[source_name]['spectra']
    peaks = sources_data[source_name]['peaks']
    
    n_channels, n_detectors = spectrum.shape
    channels = np.arange(n_channels)
    
    # Filter out poor detectors
    good_detectors = [det for det in range(n_detectors) 
                      if det not in poor_detectors]
    spectrum_filtered = spectrum[:, good_detectors].copy()
    
    # Apply count threshold
    spectrum_filtered[spectrum_filtered < count_threshold] = 0
    
    # Create energy matrix
    energy_matrix = np.zeros_like(spectrum_filtered, dtype=float)
    for i, det in enumerate(good_detectors):
        if det in calibration_results:
            calib = calibration_results[det]['calibration']
            if calib['slope'] is not None:
                energy_matrix[:, i] = channel_to_energy(
                    channels, calib['slope'], calib['intercept']
                )
    
    # Convert peaks to energy space
    converted_peaks = []
    for det in good_detectors:
        if det in peaks and det in calibration_results:
            calib = calibration_results[det]['calibration']
            if calib['slope'] is not None:
                new_det_idx = good_detectors.index(det)
                # Handle new peak structure
                peak_data = peaks[det]
                if isinstance(peak_data, dict) and 'calibration_peaks' in peak_data:
                    detector_peaks = peak_data['calibration_peaks']
                else:
                    detector_peaks = peak_data
                    
                for peak in detector_peaks:
                    energy = channel_to_energy(
                        peak['peak_index'], calib['slope'], calib['intercept']
                    )
                    converted_peaks.append([energy, new_det_idx, peak['persistence']])
    
    converted_peaks = np.array(converted_peaks) if converted_peaks else np.array([]).reshape(0, 3)
    
    # 3D surface plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    detector_mesh = np.tile(np.arange(len(good_detectors)), (n_channels, 1))
    
    surf = ax.plot_surface(energy_matrix, detector_mesh, spectrum_filtered,
                           cmap='viridis', edgecolor='none', alpha=0.8)
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Counts')
    
    if len(converted_peaks) > 0:
        ax.scatter(converted_peaks[:, 0], converted_peaks[:, 1],
                   converted_peaks[:, 2] * np.max(spectrum_filtered),
                   c='red', marker='o', s=60, edgecolor='k',
                   label="Detected Peaks", depthshade=True)
    
    ax.set_xlabel('Energy (MeV)', fontsize=FONTSIZE["label"])
    ax.set_ylabel('Detector', fontsize=FONTSIZE["label"])
    ax.set_zlabel('Counts', fontsize=FONTSIZE["label"])
    ax.set_title(f'3D Spectrum - {source_name}', fontsize=FONTSIZE["title"])
    
    if len(converted_peaks) > 0:
        ax.legend(fontsize=FONTSIZE["legend"])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_2d_energy_heatmap(sources_data: Dict[str, Dict],
                           calibration_results: Dict[int, Dict],
                           source_name: str,
                           poor_detectors: List[int],
                           count_threshold: int = 100,
                           save_path: Optional[str] = None):
    """
    Create 2D heatmap of spectrum in energy space.
    
    Parameters:
        sources_data (Dict[str, Dict]): Data from all sources
        calibration_results (Dict[int, Dict]): Calibration results
        source_name (str): Source to visualize
        poor_detectors (List[int]): Detectors to exclude
        count_threshold (int): Minimum count threshold for filtering
        save_path (Optional[str]): Path to save plot
    """
    # Custom colormap
    colors = ['white', '#667DDC', '#EBFBE9', '#F92533', '#DE0223']
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
    
    if source_name not in sources_data:
        print(f"Source {source_name} not found in data")
        return
    
    spectrum = sources_data[source_name]['spectra']
    peaks = sources_data[source_name]['peaks']
    
    n_channels, n_detectors = spectrum.shape
    channels = np.arange(n_channels)
    
    # Filter out poor detectors
    good_detectors = [det for det in range(n_detectors) 
                      if det not in poor_detectors]
    spectrum_filtered = spectrum[:, good_detectors].copy()
    
    # Apply count threshold
    spectrum_filtered[spectrum_filtered < count_threshold] = 0
    
    # Create energy matrix and flatten for scatter plot
    energy_values = []
    detector_values = []
    count_values = []
    
    for i, det in enumerate(good_detectors):
        if det in calibration_results:
            calib = calibration_results[det]['calibration']
            if calib['slope'] is not None:
                energies = channel_to_energy(
                    channels, calib['slope'], calib['intercept']
                )
                energy_values.extend(energies)
                detector_values.extend([i] * len(energies))
                count_values.extend(spectrum_filtered[:, i])
    
    # Convert peaks to energy space
    converted_peaks = []
    for det in good_detectors:
        if det in peaks and det in calibration_results:
            calib = calibration_results[det]['calibration']
            if calib['slope'] is not None:
                new_det_idx = good_detectors.index(det)
                for peak in peaks[det]:
                    energy = channel_to_energy(
                        peak['peak_index'], calib['slope'], calib['intercept']
                    )
                    converted_peaks.append([energy, new_det_idx])
    
    converted_peaks = np.array(converted_peaks) if converted_peaks else np.array([]).reshape(0, 2)
    
    # Create 2D heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sc = ax.scatter(energy_values, detector_values, c=count_values, 
                    cmap=cmap, marker='s', s=1, alpha=0.8)
    
    cbar = fig.colorbar(sc)
    cbar.set_label('Counts', fontsize=FONTSIZE["label"])
    cbar.ax.tick_params(labelsize=FONTSIZE["ticks"])
    
    # Overlay peaks
    if len(converted_peaks) > 0:
        ax.scatter(converted_peaks[:, 0], converted_peaks[:, 1],
                   facecolors='#BF2828', edgecolors='k', s=50,
                   label="Detected Peaks", marker='o', linewidth=1, alpha=0.9)
        ax.legend(fontsize=FONTSIZE["legend"])
    
    # Determine energy range
    if energy_values:
        energy_range = (min(energy_values), max(energy_values))
        ax.set_xlim(energy_range)
    
    ax.set_ylim(-1, len(good_detectors))
    ax.set_xlabel('Energy (MeV)', fontsize=FONTSIZE["label"])
    ax.set_ylabel('Detector', fontsize=FONTSIZE["label"])
    ax.set_title(f'2D Energy Heatmap - {source_name}', fontsize=FONTSIZE["title"])
    ax.tick_params(axis='both', labelsize=FONTSIZE["ticks"])
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
