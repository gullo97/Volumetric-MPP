#!/usr/bin/env python3
"""
Interactive Calibration Fixer

This script allows users to manually correct failed detector calibrations by:
1. Loading calibration results and identifying detectors with high CV errors
2. Displaying spectra for failed detectors with interactive peak selection
3. Updating calibration parameters based on user-selected peaks
4. Exporting complete calibration results to Excel

Usage:
    python interactive_calibration_fixer.py
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from core import find_peaks_volumetric_persistence


class InteractiveCalibrationFixer:
    """Interactive GUI for fixing failed detector calibrations."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Detector Calibration Fixer")
        self.root.geometry("1400x900")
        
        # Configuration
        self.sources = {
            'Sodium': {
                'sheet_name': 'Sodio I',
                'expected_energies': [0.511],
                'color': '#FF6B6B'
            },
            'Cesium': {
                'sheet_name': 'Cesio I',
                'expected_energies': [0.662],
                'color': '#4ECDC4'
            },
            'Cobalt': {
                'sheet_name': 'Cobalto I',
                'expected_energies': [1.173, 1.332],
                'color': '#45B7D1'
            }
        }
        
        self.data_file = "../Data/Dati_luglio.xlsx"
        self.min_channel = 250
        self.cv_error_threshold = 0.001  # Default threshold for CV error
        
        # Data storage
        self.calibration_results = None
        self.failed_detectors = []
        self.current_detector_idx = 0
        self.selected_peaks = {}  # {source_name: [(channel, intensity)]}
        self.current_spectra = {}  # {source_name: (spectrum, channels)}
        
        # GUI setup
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the main GUI layout."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel",
                                       padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Load results button
        load_btn = ttk.Button(control_frame, text="Load Calibration Results",
                              command=self.load_calibration_results)
        load_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Threshold selection
        thresh_label = ttk.Label(control_frame, text="CV Error Threshold:")
        thresh_label.pack(side=tk.LEFT, padx=(10, 5))
        
        self.threshold_var = tk.DoubleVar(value=self.cv_error_threshold)
        threshold_entry = ttk.Entry(control_frame,
                                    textvariable=self.threshold_var,
                                    width=10)
        threshold_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        filter_btn = ttk.Button(control_frame, text="Filter Failed Detectors",
                                command=self.filter_failed_detectors)
        filter_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Export button
        export_btn = ttk.Button(control_frame, text="Export Results to Excel",
                                command=self.export_to_excel)
        export_btn.pack(side=tk.RIGHT)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(main_frame, text="Progress",
                                        padding="10")
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_label = ttk.Label(progress_frame,
                                        text="Load calibration results to start")
        self.progress_label.pack(side=tk.LEFT)
        
        self.detector_label = ttk.Label(progress_frame, text="")
        self.detector_label.pack(side=tk.RIGHT)
        
        # Navigation frame
        nav_frame = ttk.Frame(progress_frame)
        nav_frame.pack(fill=tk.X, pady=(10, 0))
        
        prev_btn = ttk.Button(nav_frame, text="Previous Detector",
                              command=self.previous_detector)
        prev_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        next_btn = ttk.Button(nav_frame, text="Next Detector",
                              command=self.next_detector)
        next_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        skip_btn = ttk.Button(nav_frame, text="Skip Detector",
                              command=self.skip_detector)
        skip_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        apply_btn = ttk.Button(nav_frame, text="Apply Calibration",
                               command=self.apply_calibration)
        apply_btn.pack(side=tk.RIGHT)
        
        # Main content area
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Peak selection info
        left_panel = ttk.LabelFrame(content_frame, text="Selected Peaks",
                                    padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Peak selection display
        self.peak_info_text = tk.Text(left_panel, width=30, height=20,
                                      wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(left_panel, orient=tk.VERTICAL,
                                  command=self.peak_info_text.yview)
        self.peak_info_text.configure(yscrollcommand=scrollbar.set)
        self.peak_info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Clear selections button
        clear_btn = ttk.Button(left_panel, text="Clear All Selections",
                               command=self.clear_selections)
        clear_btn.pack(pady=(10, 0))
        
        # Right panel - Spectrum plots
        plot_frame = ttk.LabelFrame(content_frame, text="Detector Spectra",
                                    padding="10")
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 12))
        self.fig.suptitle("Click on peaks to select them for calibration",
                          fontsize=14)
        
        # Embed matplotlib in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Connect click events
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)
        
        # Navigation toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        
    def load_calibration_results(self):
        """Load calibration results from the notebook output."""
        try:
            # Load the calibration results DataFrame
            file_path = filedialog.askopenfilename(
                title="Select Calibration Results CSV",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialdir=os.path.dirname(__file__)
            )
            
            if not file_path:
                return
                
            self.calibration_results = pd.read_csv(file_path)
            
            # Also try to load the raw results if available
            self.load_raw_results()
            
            msg = f"Loaded calibration results for {len(self.calibration_results)} detectors"
            messagebox.showinfo("Success", msg)
            self.filter_failed_detectors()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load calibration results: {e}")
    
    def load_raw_results(self):
        """Try to load raw calibration results for more detailed analysis."""
        try:
            json_path = os.path.join(os.path.dirname(__file__),
                                     'detector_calibration_functions.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    self.raw_calibration_data = json.load(f)
        except Exception:
            self.raw_calibration_data = {}
    
    def filter_failed_detectors(self):
        """Filter detectors that need manual correction based on CV error threshold."""
        if self.calibration_results is None:
            messagebox.showwarning("Warning", "Please load calibration results first")
            return
        
        threshold = self.threshold_var.get()
        
        # Find detectors with CV RMSE above threshold or missing calibration
        failed_mask = ((self.calibration_results['CV_RMSE_MeV'] > threshold) |
                       (self.calibration_results['CV_RMSE_MeV'].isna()) |
                       (self.calibration_results['Num_Calibration_Points'] < 3))
        
        self.failed_detectors = (self.calibration_results[failed_mask]['Detector']
                                 .tolist())
        
        if not self.failed_detectors:
            msg = "No detectors need manual correction with current threshold"
            messagebox.showinfo("Info", msg)
            return
        
        self.current_detector_idx = 0
        msg = f"Found {len(self.failed_detectors)} detectors needing correction"
        self.progress_label.config(text=msg)
        self.load_current_detector()
    
    def load_current_detector(self):
        """Load spectra for the current detector."""
        if not self.failed_detectors:
            return
        
        detector_id = self.failed_detectors[self.current_detector_idx]
        text = (f"Detector {detector_id} "
                f"({self.current_detector_idx + 1}/{len(self.failed_detectors)})")
        self.detector_label.config(text=text)
        
        # Clear previous selections
        self.selected_peaks = {}
        self.current_spectra = {}
        
        # Load spectra for each source
        for source_name, config in self.sources.items():
            try:
                spectrum, channels = self.load_detector_spectrum(
                    detector_id, config['sheet_name'])
                self.current_spectra[source_name] = (spectrum, channels)
            except Exception as e:
                print(f"Error loading {source_name} spectrum for "
                      f"detector {detector_id}: {e}")
        
        self.plot_spectra()
        self.update_peak_info()
    
    def load_detector_spectrum(self, detector_idx, sheet_name):
        """Load and preprocess a detector spectrum."""
        df = pd.read_excel(self.data_file, sheet_name=sheet_name)
        spectrum = df.iloc[:, detector_idx].to_numpy()
        
        # Apply channel filtering
        filtered_spectrum = spectrum[self.min_channel:]
        channels = np.arange(self.min_channel, len(spectrum))
        
        # Normalize by area
        normalized_spectrum = filtered_spectrum / np.trapz(filtered_spectrum)
        
        return normalized_spectrum, channels
    
    def plot_spectra(self):
        """Plot spectra for all sources with detected peaks."""
        for ax in self.axes:
            ax.clear()
        
        for i, (source_name, config) in enumerate(self.sources.items()):
            if source_name not in self.current_spectra:
                continue
                
            spectrum, channels = self.current_spectra[source_name]
            ax = self.axes[i]
            
            # Plot spectrum
            ax.plot(channels, spectrum, color='#333333', linewidth=1, alpha=0.8)
            
            # Show automatically detected peaks using volumetric persistence
            try:
                peaks_info = find_peaks_volumetric_persistence(
                    spectrum,
                    smoothing_range=[1, 3, 5],
                    bins_factor_range=[1, 2],
                    threshold_range=np.linspace(0, 0.004, 7),
                    width_range=np.linspace(1, 50, 7),
                    prominence_range=np.linspace(0.0001, 0.005, 7),
                    distance_range=np.array([1, 5, 10, 15, 20]),
                    top_k=8,
                    parallel=True
                )
                
                # Show top detected peaks as suggestions
                for j, peak in enumerate(peaks_info[:5]):
                    peak_channel = peak['peak_index'] + self.min_channel
                    ax.axvline(peak_channel, color='lightblue',
                               linestyle='--', alpha=0.5, linewidth=1)
                    if j < 3:  # Label only top 3
                        ax.text(peak_channel, ax.get_ylim()[1] * 0.9,
                                f'Auto {j+1}', rotation=90, ha='right',
                                va='top', fontsize=8, color='blue')
                
            except Exception as e:
                print(f"Error detecting peaks for {source_name}: {e}")
            
            # Highlight selected peaks
            if source_name in self.selected_peaks:
                for channel, _ in self.selected_peaks[source_name]:
                    ax.axvline(channel, color=config['color'],
                               linewidth=3, alpha=0.8)
                    spectrum_idx = channel - self.min_channel
                    ax.scatter(channel, spectrum[spectrum_idx],
                               color=config['color'], s=100, zorder=5,
                               edgecolor='black')
            
            # Labels and formatting
            expected_energies = config['expected_energies']
            energy_str = ', '.join(f'{e:.3f}' for e in expected_energies)
            ax.set_title(f'{source_name} - Expected: {energy_str} MeV',
                         fontsize=12)
            ax.set_ylabel('Normalized Intensity')
            ax.grid(True, alpha=0.3)
            
            if i == len(self.sources) - 1:
                ax.set_xlabel('Channel')
        
        self.canvas.draw()
    
    def on_plot_click(self, event):
        """Handle mouse clicks on spectrum plots."""
        if event.inaxes is None or event.button != 1:  # Only left clicks
            return
        
        # Determine which source was clicked
        source_names = list(self.sources.keys())
        subplot_idx = None
        
        for i, ax in enumerate(self.axes):
            if event.inaxes == ax:
                subplot_idx = i
                break
        
        if subplot_idx is None or subplot_idx >= len(source_names):
            return
        
        source_name = source_names[subplot_idx]
        clicked_channel = int(round(event.xdata))
        
        if source_name not in self.current_spectra:
            return
        
        spectrum, channels = self.current_spectra[source_name]
        
        # Check if channel is valid
        if clicked_channel < channels[0] or clicked_channel > channels[-1]:
            return
        
        # Get the spectrum intensity at clicked point
        spectrum_idx = clicked_channel - self.min_channel
        intensity = spectrum[spectrum_idx]
        
        # Initialize selected peaks for this source if needed
        if source_name not in self.selected_peaks:
            self.selected_peaks[source_name] = []
        
        # Check if click is near an existing peak (within 10 channels)
        existing_peak_idx = None
        for i, (channel, _) in enumerate(self.selected_peaks[source_name]):
            if abs(channel - clicked_channel) <= 10:
                existing_peak_idx = i
                break
        
        if existing_peak_idx is not None:
            # Remove existing peak
            self.selected_peaks[source_name].pop(existing_peak_idx)
        else:
            # Add new peak
            expected_count = len(self.sources[source_name]['expected_energies'])
            if len(self.selected_peaks[source_name]) < expected_count:
                self.selected_peaks[source_name].append((clicked_channel,
                                                         intensity))
            else:
                msg = (f"{source_name} expects only {expected_count} peak(s). "
                       "Remove existing peaks first.")
                messagebox.showwarning("Warning", msg)
                return
        
        self.plot_spectra()
        self.update_peak_info()
    
    def update_peak_info(self):
        """Update the peak selection information display."""
        self.peak_info_text.delete(1.0, tk.END)
        
        detector_id = (self.failed_detectors[self.current_detector_idx]
                       if self.failed_detectors else "N/A")
        self.peak_info_text.insert(tk.END, f"Detector {detector_id}\n")
        self.peak_info_text.insert(tk.END, "=" * 20 + "\n\n")
        
        total_peaks = 0
        total_expected = 0
        
        for source_name, config in self.sources.items():
            expected_energies = config['expected_energies']
            total_expected += len(expected_energies)
            
            self.peak_info_text.insert(tk.END, f"{source_name}:\n")
            self.peak_info_text.insert(tk.END, f"Expected: {expected_energies}\n")
            
            if (source_name in self.selected_peaks
                    and self.selected_peaks[source_name]):
                total_peaks += len(self.selected_peaks[source_name])
                self.peak_info_text.insert(tk.END, "Selected peaks:\n")
                for i, (channel, intensity) in enumerate(
                        self.selected_peaks[source_name]):
                    text = f"  Peak {i+1}: Channel {channel:.0f}\n"
                    self.peak_info_text.insert(tk.END, text)
            else:
                self.peak_info_text.insert(tk.END, "No peaks selected\n")
            
            self.peak_info_text.insert(tk.END, "\n")
        
        # Summary
        self.peak_info_text.insert(tk.END,
                                   f"Total: {total_peaks}/{total_expected} peaks selected\n")
        
        if total_peaks >= 2:
            self.peak_info_text.insert(tk.END, "\nReady for calibration!\n")
        else:
            self.peak_info_text.insert(tk.END,
                                       "\nNeed at least 2 peaks for calibration\n")
    
    def clear_selections(self):
        """Clear all peak selections."""
        self.selected_peaks = {}
        self.plot_spectra()
        self.update_peak_info()
    
    def apply_calibration(self):
        """Apply calibration using selected peaks."""
        if not self.selected_peaks:
            messagebox.showwarning("Warning", "No peaks selected")
            return
        
        # Collect calibration points
        calibration_points = []
        
        for source_name, peaks in self.selected_peaks.items():
            expected_energies = self.sources[source_name]['expected_energies']
            
            if len(peaks) != len(expected_energies):
                msg = (f"{source_name}: Expected {len(expected_energies)} peaks, "
                       f"got {len(peaks)}")
                messagebox.showwarning("Warning", msg)
                return
            
            # Sort peaks by channel and pair with sorted energies
            sorted_peaks = sorted(peaks, key=lambda x: x[0])
            sorted_energies = sorted(expected_energies)
            
            for (channel, _), energy in zip(sorted_peaks, sorted_energies):
                calibration_points.append((channel, energy))
        
        if len(calibration_points) < 2:
            messagebox.showwarning("Warning", "Need at least 2 calibration points")
            return
        
        # Perform linear calibration
        channels, energies = zip(*calibration_points)
        slope, intercept, r2, cv_error = self.perform_calibration(channels, energies)
        
        # Update results
        detector_id = self.failed_detectors[self.current_detector_idx]
        self.update_calibration_result(detector_id, slope, intercept, r2,
                                       cv_error, len(calibration_points))
        
        # Show results
        result_msg = f"Calibration applied for Detector {detector_id}:\n"
        result_msg += f"Slope: {slope:.6f} MeV/channel\n"
        result_msg += f"Intercept: {intercept:.6f} MeV\n"
        result_msg += f"R²: {r2:.4f}\n"
        result_msg += f"CV RMSE: {np.sqrt(cv_error):.6f} MeV"
        
        messagebox.showinfo("Calibration Applied", result_msg)
        
        # Move to next detector
        self.next_detector()
    
    def perform_calibration(self, channels, energies):
        """Perform linear calibration and calculate metrics."""
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import LeaveOneOut
        from sklearn.metrics import r2_score
        
        # Linear regression
        model = LinearRegression()
        X = np.array(channels).reshape(-1, 1)
        y = np.array(energies)
        
        model.fit(X, y)
        y_pred = model.predict(X)
        
        slope = model.coef_[0]
        intercept = model.intercept_
        r2 = r2_score(y, y_pred)
        
        # Cross-validation error
        if len(channels) >= 3:
            loo = LeaveOneOut()
            cv_errors = []
            
            for train_idx, test_idx in loo.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                cv_model = LinearRegression()
                cv_model.fit(X_train, y_train)
                y_cv_pred = cv_model.predict(X_test)
                
                error = (y_test[0] - y_cv_pred[0])**2
                cv_errors.append(error)
            
            cv_error = np.mean(cv_errors)
        else:
            cv_error = np.nan
        
        return slope, intercept, r2, cv_error
    
    def update_calibration_result(self, detector_id, slope, intercept, r2,
                                  cv_error, num_points):
        """Update the calibration results DataFrame."""
        if self.calibration_results is None:
            return
        
        # Find the detector in the results
        detector_mask = self.calibration_results['Detector'] == detector_id
        
        if detector_mask.any():
            # Update existing entry
            self.calibration_results.loc[detector_mask,
                                         'Slope_MeV_per_channel'] = slope
            self.calibration_results.loc[detector_mask, 'Intercept_MeV'] = intercept
            self.calibration_results.loc[detector_mask, 'R2_Score'] = r2
            cv_rmse = np.sqrt(cv_error) if not np.isnan(cv_error) else np.nan
            self.calibration_results.loc[detector_mask, 'CV_RMSE_MeV'] = cv_rmse
            self.calibration_results.loc[detector_mask,
                                         'Num_Calibration_Points'] = num_points
        else:
            # Add new entry
            cv_rmse = np.sqrt(cv_error) if not np.isnan(cv_error) else np.nan
            new_row = pd.DataFrame({
                'Detector': [detector_id],
                'Slope_MeV_per_channel': [slope],
                'Intercept_MeV': [intercept],
                'R2_Score': [r2],
                'CV_RMSE_MeV': [cv_rmse],
                'Num_Calibration_Points': [num_points]
            })
            self.calibration_results = pd.concat([self.calibration_results,
                                                  new_row], ignore_index=True)
    
    def previous_detector(self):
        """Move to previous detector."""
        if self.current_detector_idx > 0:
            self.current_detector_idx -= 1
            self.load_current_detector()
    
    def next_detector(self):
        """Move to next detector."""
        if self.current_detector_idx < len(self.failed_detectors) - 1:
            self.current_detector_idx += 1
            self.load_current_detector()
        else:
            messagebox.showinfo("Complete", "All detectors have been processed!")
    
    def skip_detector(self):
        """Skip current detector without applying calibration."""
        self.next_detector()
    
    def export_to_excel(self):
        """Export complete calibration results to Excel."""
        if self.calibration_results is None:
            messagebox.showwarning("Warning", "No calibration results to export")
            return
        
        # Create complete results DataFrame with all 128 detectors
        complete_results = pd.DataFrame({
            'Detector': range(128),
            'Slope_MeV_per_channel': np.nan,
            'Intercept_MeV': np.nan,
            'R2_Score': np.nan,
            'CV_RMSE_MeV': np.nan,
            'Num_Calibration_Points': 0,
            'Calibration_Status': 'Failed'
        })
        
        # Update with actual calibration results
        for _, row in self.calibration_results.iterrows():
            detector_id = int(row['Detector'])
            if 0 <= detector_id < 128:
                complete_results.loc[detector_id,
                                     'Slope_MeV_per_channel'] = row['Slope_MeV_per_channel']
                complete_results.loc[detector_id, 'Intercept_MeV'] = row['Intercept_MeV']
                complete_results.loc[detector_id, 'R2_Score'] = row['R2_Score']
                complete_results.loc[detector_id, 'CV_RMSE_MeV'] = row['CV_RMSE_MeV']
                complete_results.loc[detector_id,
                                     'Num_Calibration_Points'] = row['Num_Calibration_Points']
                
                # Determine status
                has_slope = pd.notna(row['Slope_MeV_per_channel'])
                has_points = row['Num_Calibration_Points'] >= 2
                if has_slope and has_points:
                    has_cv = pd.notna(row['CV_RMSE_MeV'])
                    if has_cv and row['CV_RMSE_MeV'] <= 0.01:
                        complete_results.loc[detector_id, 'Calibration_Status'] = 'Good'
                    else:
                        complete_results.loc[detector_id, 'Calibration_Status'] = 'Acceptable'
                else:
                    complete_results.loc[detector_id, 'Calibration_Status'] = 'Failed'
        
        # Ask for export file
        file_path = filedialog.asksaveasfilename(
            title="Save Complete Calibration Results",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"),
                       ("All files", "*.*")],
            initialdir=os.path.dirname(__file__)
        )
        
        if not file_path:
            return
        
        try:
            if file_path.endswith('.xlsx'):
                # Export to Excel with multiple sheets
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    complete_results.to_excel(writer,
                                              sheet_name='Complete_Calibration',
                                              index=False)
                    
                    # Summary statistics
                    successful = complete_results[
                        complete_results['Calibration_Status'] != 'Failed']
                    summary = pd.DataFrame({
                        'Metric': ['Total Detectors', 'Successfully Calibrated',
                                   'Success Rate (%)',
                                   'Mean Slope (MeV/channel)', 'Std Slope',
                                   'Mean Intercept (MeV)', 'Std Intercept',
                                   'Mean R²', 'Mean CV RMSE (MeV)'],
                        'Value': [
                            128,
                            len(successful),
                            100 * len(successful) / 128,
                            successful['Slope_MeV_per_channel'].mean(),
                            successful['Slope_MeV_per_channel'].std(),
                            successful['Intercept_MeV'].mean(),
                            successful['Intercept_MeV'].std(),
                            successful['R2_Score'].mean(),
                            successful['CV_RMSE_MeV'].mean()
                        ]
                    })
                    summary.to_excel(writer, sheet_name='Summary_Statistics',
                                     index=False)
            else:
                complete_results.to_csv(file_path, index=False)
            
            messagebox.showinfo("Success", f"Results exported to {file_path}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results: {e}")


def main():
    """Main application entry point."""
    root = tk.Tk()
    app = InteractiveCalibrationFixer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
