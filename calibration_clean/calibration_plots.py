"""
Calibration visualization functions for detector calibration analysis.

This module provides clean plotting functions for visualizing calibration
results and spectral data.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Optional
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration_core import channel_to_energy

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
    """Create overview plots of calibration quality across all detectors."""
    # Extract data for plotting
    detectors, slopes, intercepts, cv_errors = [], [], [], []
    
    for det, result in calibration_results.items():
        calib = result.get('calibration', {})
        if (calib.get('slope') is not None and
                calib.get('cv_error') is not None):
            detectors.append(det)
            slopes.append(calib['slope'])
            intercepts.append(calib['intercept'])
            cv_errors.append(calib['cv_error'])
    
    detectors = np.array(detectors)
    slopes = np.array(slopes)
    intercepts = np.array(intercepts)
    cv_errors = np.array(cv_errors)
    
    # Check if we have any valid data
    if len(cv_errors) == 0:
        print("⚠️ No valid calibration data with CV errors found for plotting")
        return
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Histogram of CV errors
    axes[0, 0].hist(cv_errors, bins=min(20, len(cv_errors)), color='skyblue',
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
    
    # 5. R² distribution
    r_squareds = [result['calibration']['r_squared'] 
                  for result in calibration_results.values()
                  if result['calibration']['r_squared'] is not None]
    
    axes[1, 1].hist(r_squareds, bins=20, color='lightcoral', 
                    edgecolor='k', alpha=0.85)
    axes[1, 1].set_title('R² Distribution', fontsize=FONTSIZE["title"])
    axes[1, 1].set_xlabel('R²', fontsize=FONTSIZE["label"])
    axes[1, 1].set_ylabel('Frequency', fontsize=FONTSIZE["label"])
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)
    axes[1, 1].tick_params(axis='both', labelsize=FONTSIZE["ticks"])
    
    # 6. Summary statistics
    axes[1, 2].axis('off')
    stats_text = f"""
    Calibration Statistics:
    
    Total Detectors: {len(calibration_results)}
    Successful: {len(detectors)}
    
    CV Error (MeV):
      Mean: {np.mean(cv_errors):.4f}
      Std:  {np.std(cv_errors):.4f}
      Min:  {np.min(cv_errors):.4f}
      Max:  {np.max(cv_errors):.4f}
    
    R²:
      Mean: {np.mean(r_squareds):.4f}
      Min:  {np.min(r_squareds):.4f}
      Max:  {np.max(r_squareds):.4f}
    """
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=12, 
                     verticalalignment='center',
                     fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_single_detector_calibration(detector_idx: int, 
                                    calibration_result: Dict,
                                    save_path: Optional[str] = None):
    """Plot calibration curve for a single detector."""
    from scipy.stats import linregress
    
    calibration = calibration_result['calibration']
    peaks_per_source = calibration_result['peaks_per_source']
    
    # Collect calibration points
    channels = []
    energies = []
    source_labels = []
    
    # This would need expected_energies passed in, simplified for now
    source_energies = {
        'Sodium': [0.511],
        'Cesium': [0.662], 
        'Cobalt': [1.17, 1.33]
    }
    
    for source_name, peak_channels in peaks_per_source.items():
        if source_name in source_energies:
            source_energy_list = source_energies[source_name]
            n_matches = min(len(peak_channels), len(source_energy_list))
            
            for i in range(n_matches):
                channels.append(peak_channels[i])
                energies.append(source_energy_list[i])
                source_labels.append(source_name)
    
    if len(channels) < 2:
        print(f"Detector {detector_idx}: Insufficient calibration points")
        return
    
    channels = np.array(channels)
    energies = np.array(energies)
    
    # Calculate LOOCV errors for error bars
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
    x_range = np.linspace(min(channels)*0.8, max(channels)*1.2, 100)
    y_range = slope * x_range + intercept
    
    plt.figure(figsize=(10, 6))
    
    # Color points by source
    colors = {'Sodium': 'green', 'Cesium': 'orange', 'Cobalt': 'red'}
    for i, (ch, en, src) in enumerate(zip(channels, energies, source_labels)):
        color = colors.get(src, 'blue')
        plt.errorbar(ch, en, yerr=loocv_errors[i], fmt='o', 
                     color=color, capsize=5, markersize=8,
                     markerfacecolor='white', markeredgecolor=color,
                     label=src if src not in [source_labels[j] for j in range(i)] else "")
    
    line_label = f'Fit: E = {slope:.4f}×Ch + {intercept:.4f}'
    plt.plot(x_range, y_range, label=line_label, color='black', 
             linestyle='-', linewidth=2)
    
    plt.xlabel('Channel', fontsize=FONTSIZE["label"])
    plt.ylabel('Energy [MeV]', fontsize=FONTSIZE["label"])
    
    # Handle None values safely in title
    r_squared = calibration["r_squared"]
    cv_error = calibration["cv_error"]
    r_squared_str = f'{r_squared:.4f}' if r_squared is not None else 'N/A'
    cv_error_str = f'{cv_error:.4f} MeV' if cv_error is not None else 'N/A'
    
    plt.title(f'Detector {detector_idx} Calibration\n'
              f'R² = {r_squared_str}, '
              f'CV Error = {cv_error_str}',
              fontsize=FONTSIZE["title"])
    plt.legend(fontsize=FONTSIZE["legend"])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tick_params(axis='both', labelsize=FONTSIZE["ticks"])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_combined_source_spectra(detector_idx: int,
                                sources_data: Dict[str, Dict],
                                source_colors: Optional[Dict[str, str]] = None,
                                save_path: Optional[str] = None):
    """Plot combined spectra from all sources for a single detector."""
    if source_colors is None:
        source_colors = {
            "Cesium": "#CBAC88", 
            "Sodium": "#69995D", 
            "Cobalt": "#C94040"
        }
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Combined spectra
    for source_name, source_data in sources_data.items():
        spectrum = source_data['spectra'][:, detector_idx]
        channels = np.arange(len(spectrum))
        color = source_colors.get(source_name, 'blue')
        
        ax1.plot(channels, spectrum, label=source_name, 
                color=color, linewidth=1.5, alpha=0.8)
        
        # Mark detected peaks
        peaks_data = source_data['peaks'].get(detector_idx, {})
        if isinstance(peaks_data, dict) and 'calibration_peaks' in peaks_data:
            calibration_peaks = peaks_data['calibration_peaks']
        else:
            calibration_peaks = peaks_data if isinstance(peaks_data, list) else []
        
        for peak_ch in calibration_peaks:
            if 0 <= peak_ch < len(spectrum):
                ax1.axvline(peak_ch, color=color, linestyle='--', alpha=0.7)
                ax1.plot(peak_ch, spectrum[peak_ch], 'o', 
                        color=color, markersize=8, markerfacecolor='white',
                        markeredgewidth=2)
    
    ax1.set_xlabel('Channel', fontsize=FONTSIZE["label"])
    ax1.set_ylabel('Counts', fontsize=FONTSIZE["label"])
    ax1.set_title(f'Detector {detector_idx} - Combined Source Spectra', 
                 fontsize=FONTSIZE["title"])
    ax1.legend(fontsize=FONTSIZE["legend"])
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=FONTSIZE["ticks"])
    
    # Plot 2: Peak detection barcode
    ax2.set_xlim(ax1.get_xlim())
    y_pos = 0
    
    for source_name, source_data in sources_data.items():
        color = source_colors.get(source_name, 'blue')
        peaks_data = source_data['peaks'].get(detector_idx, {})
        
        # Show all detected peaks
        if isinstance(peaks_data, dict) and 'all_detected_peaks' in peaks_data:
            all_peaks = peaks_data['all_detected_peaks']
        else:
            all_peaks = peaks_data if isinstance(peaks_data, list) else []
        
        # Show calibration peaks
        if isinstance(peaks_data, dict) and 'calibration_peaks' in peaks_data:
            calibration_peaks = peaks_data['calibration_peaks']
        else:
            calibration_peaks = peaks_data if isinstance(peaks_data, list) else []
        
        # Plot all detected peaks as thin lines
        for peak in all_peaks:
            ax2.axvline(peak, color=color, alpha=0.3, linewidth=1)
        
        # Plot calibration peaks as thick lines
        for peak in calibration_peaks:
            ax2.axvline(peak, color=color, alpha=0.8, linewidth=3)
        
        # Add source label
        ax2.text(ax2.get_xlim()[1]*0.02, y_pos + 0.3, source_name, 
                color=color, fontsize=12, fontweight='bold')
        y_pos += 1
    
    ax2.set_xlabel('Channel', fontsize=FONTSIZE["label"])
    ax2.set_ylabel('Source', fontsize=FONTSIZE["label"])
    ax2.set_title('Peak Detection Results (Thin=All Detected, Thick=Calibration)', 
                 fontsize=14)
    ax2.set_ylim(-0.5, len(sources_data) - 0.5)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=FONTSIZE["ticks"])
    
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
    """Create 3D visualization of energy spectrum."""
    from mpl_toolkits.mplot3d import Axes3D
    
    if source_name not in sources_data:
        print(f"Source {source_name} not found in data")
        return
    
    spectra = sources_data[source_name]['spectra']
    n_channels, n_detectors = spectra.shape
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create energy axis for calibrated detectors
    channels = np.arange(n_channels)
    
    for detector_idx in range(0, min(n_detectors, 50), 2):  # Sample detectors
        spectrum = spectra[:, detector_idx]
        
        # Skip if too noisy
        if np.max(spectrum) < count_threshold:
            continue
        
        # Convert to energy if calibrated
        if (detector_idx in calibration_results and 
            calibration_results[detector_idx]['calibration']['slope'] is not None):
            
            calib = calibration_results[detector_idx]['calibration']
            energies = channel_to_energy(channels, calib['slope'], calib['intercept'])
            x_axis = energies
            x_label = 'Energy [MeV]'
        else:
            x_axis = channels
            x_label = 'Channel'
        
        # Color based on detector quality
        color = 'red' if detector_idx in poor_detectors else 'blue'
        alpha = 0.6 if detector_idx in poor_detectors else 0.8
        
        ax.plot(x_axis, [detector_idx] * len(x_axis), spectrum,
               color=color, alpha=alpha, linewidth=1)
    
    ax.set_xlabel(x_label, fontsize=FONTSIZE["label"])
    ax.set_ylabel('Detector Index', fontsize=FONTSIZE["label"])
    ax.set_zlabel('Counts', fontsize=FONTSIZE["label"])
    ax.set_title(f'3D Energy Spectrum - {source_name}', fontsize=FONTSIZE["title"])
    
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
    """Create 2D heatmap of energy spectrum."""
    if source_name not in sources_data:
        print(f"Source {source_name} not found in data")
        return
    
    spectra = sources_data[source_name]['spectra']
    n_channels, n_detectors = spectra.shape
    
    # Filter out low-count spectra
    valid_detectors = []
    valid_spectra = []
    
    for detector_idx in range(n_detectors):
        spectrum = spectra[:, detector_idx]
        if np.max(spectrum) >= count_threshold:
            valid_detectors.append(detector_idx)
            valid_spectra.append(spectrum)
    
    if not valid_spectra:
        print("No spectra meet the count threshold")
        return
    
    spectra_array = np.array(valid_spectra).T
    
    plt.figure(figsize=(12, 8))
    
    # Use log scale for better visualization
    log_spectra = np.log10(spectra_array + 1)
    
    im = plt.imshow(log_spectra, aspect='auto', cmap='viridis',
                    extent=[0, len(valid_detectors), n_channels, 0])
    
    plt.colorbar(im, label='log₁₀(Counts + 1)')
    plt.xlabel('Detector Index', fontsize=FONTSIZE["label"])
    plt.ylabel('Channel', fontsize=FONTSIZE["label"])
    plt.title(f'2D Energy Heatmap - {source_name}', fontsize=FONTSIZE["title"])
    
    # Mark poor detectors
    for i, det_idx in enumerate(valid_detectors):
        if det_idx in poor_detectors:
            plt.axvline(i, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
