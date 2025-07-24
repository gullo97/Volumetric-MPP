#!/usr/bin/env python3
"""
Web-based Interactive Calibration Fixer

This script provides a web interface for manually correcting failed detector calibrations:
1. Loading calibration results and identifying detectors with high CV errors
2. Displaying spectra for failed detectors with interactive peak selection
3. Updating calibration parameters based on user-selected peaks
4. Exporting complete calibration results to Excel

Usage:
    python web_interactive_calibration_fixer.py
    Then open http://localhost:5000 in your web browser
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename
import tempfile
from datetime import datetime

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from core import find_peaks_volumetric_persistence

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change this in production

class WebCalibrationFixer:
    """Web-based Interactive GUI for fixing failed detector calibrations."""
    
    def __init__(self):
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
        self.raw_calibration_data = {}

# Global instance
fixer = WebCalibrationFixer()

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/upload_calibration', methods=['POST'])
def upload_calibration():
    """Handle calibration results file upload."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and file.filename.endswith('.csv'):
            # Read the CSV file
            fixer.calibration_results = pd.read_csv(file)
            
            # Try to load raw results if available
            try:
                json_path = os.path.join(os.path.dirname(__file__),
                                         'detector_calibration_functions.json')
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        fixer.raw_calibration_data = json.load(f)
            except Exception:
                fixer.raw_calibration_data = {}
            
            return jsonify({
                'success': True,
                'message': f'Loaded calibration results for {len(fixer.calibration_results)} detectors'
            })
        else:
            return jsonify({'error': 'Please upload a CSV file'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Failed to load calibration results: {str(e)}'}), 500

@app.route('/filter_failed', methods=['POST'])
def filter_failed():
    """Filter failed detectors based on threshold."""
    try:
        data = request.get_json()
        threshold = float(data.get('threshold', 0.001))
        
        if fixer.calibration_results is None:
            return jsonify({'error': 'Please load calibration results first'}), 400
        
        # Find detectors with CV RMSE above threshold or missing calibration
        failed_mask = ((fixer.calibration_results['CV_RMSE_MeV'] > threshold) |
                       (fixer.calibration_results['CV_RMSE_MeV'].isna()) |
                       (fixer.calibration_results['Num_Calibration_Points'] < 3))
        
        fixer.failed_detectors = (fixer.calibration_results[failed_mask]['Detector'].tolist())
        
        # Store in session
        session['failed_detectors'] = fixer.failed_detectors
        session['current_detector_idx'] = 0
        session['selected_peaks'] = {}
        
        return jsonify({
            'success': True,
            'failed_count': len(fixer.failed_detectors),
            'failed_detectors': fixer.failed_detectors
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to filter detectors: {str(e)}'}), 500

@app.route('/get_detector_data/<int:detector_idx>')
def get_detector_data(detector_idx):
    """Get spectra data for a specific detector."""
    try:
        # Load spectra for each source
        spectra_data = {}
        
        for source_name, config in fixer.sources.items():
            try:
                spectrum, channels = load_detector_spectrum(detector_idx, config['sheet_name'])
                
                # Get automatically detected peaks
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
                
                # Convert to serializable format
                auto_peaks = []
                for j, peak in enumerate(peaks_info[:5]):
                    peak_channel = int(peak['peak_index'] + fixer.min_channel)
                    auto_peaks.append({
                        'channel': peak_channel,
                        'intensity': float(spectrum[peak['peak_index']]),
                        'rank': j + 1
                    })
                
                spectra_data[source_name] = {
                    'channels': channels.tolist(),
                    'spectrum': spectrum.tolist(),
                    'expected_energies': config['expected_energies'],
                    'color': config['color'],
                    'auto_peaks': auto_peaks
                }
                
            except Exception as e:
                print(f"Error loading {source_name} spectrum for detector {detector_idx}: {e}")
                continue
        
        return jsonify({
            'success': True,
            'spectra': spectra_data,
            'detector_id': detector_idx
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to load detector data: {str(e)}'}), 500

def load_detector_spectrum(detector_idx, sheet_name):
    """Load and preprocess a detector spectrum."""
    df = pd.read_excel(fixer.data_file, sheet_name=sheet_name)
    spectrum = df.iloc[:, detector_idx].to_numpy()
    
    # Apply channel filtering
    filtered_spectrum = spectrum[fixer.min_channel:]
    channels = np.arange(fixer.min_channel, len(spectrum))
    
    # Normalize by area
    normalized_spectrum = filtered_spectrum / np.trapz(filtered_spectrum)
    
    return normalized_spectrum, channels

@app.route('/apply_calibration', methods=['POST'])
def apply_calibration():
    """Apply calibration using selected peaks."""
    try:
        data = request.get_json()
        detector_id = data['detector_id']
        selected_peaks = data['selected_peaks']
        
        if not selected_peaks:
            return jsonify({'error': 'No peaks selected'}), 400
        
        # Collect calibration points
        calibration_points = []
        
        for source_name, peaks in selected_peaks.items():
            if source_name not in fixer.sources:
                continue
                
            expected_energies = fixer.sources[source_name]['expected_energies']
            
            if len(peaks) != len(expected_energies):
                return jsonify({
                    'error': f"{source_name}: Expected {len(expected_energies)} peaks, got {len(peaks)}"
                }), 400
            
            # Sort peaks by channel and pair with sorted energies
            sorted_peaks = sorted(peaks, key=lambda x: x['channel'])
            sorted_energies = sorted(expected_energies)
            
            for peak, energy in zip(sorted_peaks, sorted_energies):
                calibration_points.append((peak['channel'], energy))
        
        if len(calibration_points) < 2:
            return jsonify({'error': 'Need at least 2 calibration points'}), 400
        
        # Perform linear calibration
        channels, energies = zip(*calibration_points)
        slope, intercept, r2, cv_error = perform_calibration(channels, energies)
        
        # Update results
        update_calibration_result(detector_id, slope, intercept, r2, cv_error, len(calibration_points))
        
        return jsonify({
            'success': True,
            'slope': slope,
            'intercept': intercept,
            'r2': r2,
            'cv_rmse': np.sqrt(cv_error) if not np.isnan(cv_error) else None,
            'num_points': len(calibration_points)
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to apply calibration: {str(e)}'}), 500

def perform_calibration(channels, energies):
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

def update_calibration_result(detector_id, slope, intercept, r2, cv_error, num_points):
    """Update the calibration results DataFrame."""
    if fixer.calibration_results is None:
        return
    
    # Find the detector in the results
    detector_mask = fixer.calibration_results['Detector'] == detector_id
    
    if detector_mask.any():
        # Update existing entry
        fixer.calibration_results.loc[detector_mask, 'Slope_MeV_per_channel'] = slope
        fixer.calibration_results.loc[detector_mask, 'Intercept_MeV'] = intercept
        fixer.calibration_results.loc[detector_mask, 'R2_Score'] = r2
        cv_rmse = np.sqrt(cv_error) if not np.isnan(cv_error) else np.nan
        fixer.calibration_results.loc[detector_mask, 'CV_RMSE_MeV'] = cv_rmse
        fixer.calibration_results.loc[detector_mask, 'Num_Calibration_Points'] = num_points
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
        fixer.calibration_results = pd.concat([fixer.calibration_results, new_row], ignore_index=True)

@app.route('/export_results')
def export_results():
    """Export complete calibration results to Excel."""
    try:
        if fixer.calibration_results is None:
            return jsonify({'error': 'No calibration results to export'}), 400
        
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
        for _, row in fixer.calibration_results.iterrows():
            detector_id = int(row['Detector'])
            if 0 <= detector_id < 128:
                complete_results.loc[detector_id, 'Slope_MeV_per_channel'] = row['Slope_MeV_per_channel']
                complete_results.loc[detector_id, 'Intercept_MeV'] = row['Intercept_MeV']
                complete_results.loc[detector_id, 'R2_Score'] = row['R2_Score']
                complete_results.loc[detector_id, 'CV_RMSE_MeV'] = row['CV_RMSE_MeV']
                complete_results.loc[detector_id, 'Num_Calibration_Points'] = row['Num_Calibration_Points']
                
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
        
        # Create temporary file
        temp_dir = tempfile.gettempdir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"calibration_results_{timestamp}.xlsx"
        filepath = os.path.join(temp_dir, filename)
        
        # Export to Excel with multiple sheets
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            complete_results.to_excel(writer, sheet_name='Complete_Calibration', index=False)
            
            # Summary statistics
            successful = complete_results[complete_results['Calibration_Status'] != 'Failed']
            summary = pd.DataFrame({
                'Metric': ['Total Detectors', 'Successfully Calibrated', 'Success Rate (%)',
                           'Mean Slope (MeV/channel)', 'Std Slope', 'Mean Intercept (MeV)', 'Std Intercept',
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
            summary.to_excel(writer, sheet_name='Summary_Statistics', index=False)
        
        return send_file(filepath, as_attachment=True, download_name=filename)
        
    except Exception as e:
        return jsonify({'error': f'Failed to export results: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Web Interactive Calibration Fixer...")
    print("Open http://localhost:5001 in your web browser")
    app.run(debug=True, host='0.0.0.0', port=5001)
