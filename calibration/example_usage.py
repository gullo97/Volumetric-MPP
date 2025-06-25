#!/usr/bin/env python3
"""
Example script demonstrating the new calibration system.

This script shows how to use the calibration package programmatically
without the Jupyter notebook interface.
"""

import sys
import os
import numpy as np

# Add parent directory to path to import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration_core import (
    process_source_data,
    calibrate_detector_array,
    find_poor_detectors
)
from source_config import get_source_config, print_source_info


def main():
    """Main calibration example."""
    
    print("🔬 Volumetric MPP Calibration Example")
    print("=" * 50)
    
    # Configuration
    data_file_path = "../DATI_Gullo1.xlsx/DATI_Gullo1.xlsx"
    sheet_mapping = {
        "Sodium": "Sodio I",
        "Cobalt": "Cobalto I", 
        "Cesium": "Cesio I"
    }
    
    # Show available sources
    print("\n📊 Available radioactive sources:")
    print_source_info()
    
    # Process sources
    print(f"\n🔄 Processing sources...")
    sources_data = {}
    expected_energies = {}
    
    for source_name, sheet_name in sheet_mapping.items():
        print(f"\n📡 Processing {source_name}...")
        
        try:
            # Get source configuration
            source_config = get_source_config(source_name)
            expected_energies[source_name] = source_config['energies']
            
            # Configure processing
            processing_config = {
                'sheet_name': sheet_name,
                'top_k': source_config['top_k']
            }
            
            detection_params = source_config['detection_params'].copy()
            detection_params['parallel'] = True
            
            # Process data
            spectra, peaks = process_source_data(
                data_file_path, processing_config, detection_params
            )
            
            sources_data[source_name] = {
                'spectra': spectra,
                'peaks': peaks,
                'config': source_config
            }
            
            n_channels, n_detectors = spectra.shape
            total_peaks = sum(len(detector_peaks) for detector_peaks in peaks.values())
            
            print(f"   ✅ Success: {n_channels} channels, {n_detectors} detectors")
            print(f"   🎯 Total peaks detected: {total_peaks}")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            continue
    
    if not sources_data:
        print("❌ No sources processed successfully!")
        return
    
    # Perform calibration
    print(f"\n🎯 Performing calibration...")
    
    try:
        calibration_results = calibrate_detector_array(sources_data, expected_energies)
        
        # Calculate statistics
        successful_calibrations = 0
        cv_errors = []
        
        for detector_idx, result in calibration_results.items():
            calib = result['calibration']
            if calib['slope'] is not None:
                successful_calibrations += 1
                if calib['cv_error'] is not None:
                    cv_errors.append(calib['cv_error'])
        
        print(f"✅ Calibration complete!")
        print(f"   Detectors calibrated: {successful_calibrations}/{len(calibration_results)}")
        
        if cv_errors:
            mean_cv_error = np.mean(cv_errors)
            std_cv_error = np.std(cv_errors)
            print(f"   Average CV error: {mean_cv_error:.4f} ± {std_cv_error:.4f} MeV")
        
        # Find poor detectors
        poor_detectors = find_poor_detectors(calibration_results, 0.05)
        
        print(f"\n🚨 Poor detectors (CV error > 0.05 MeV):")
        if poor_detectors:
            print(f"   Count: {len(poor_detectors)}/{len(calibration_results)}")
            print(f"   Indices: {poor_detectors[:10]}")  # Show first 10
        else:
            print(f"   None! All detectors meet quality threshold.")
        
        # Show sample results
        print(f"\n📋 Sample calibration results:")
        print("-" * 70)
        print(f"{'Det':<4} {'Slope':<12} {'Intercept':<12} {'CV Error':<10} {'R²':<8}")
        print("-" * 70)
        
        for det_idx in list(calibration_results.keys())[:10]:  # Show first 10
            result = calibration_results[det_idx]
            calib = result['calibration']
            
            slope = f"{calib['slope']:.6f}" if calib['slope'] is not None else "N/A"
            intercept = f"{calib['intercept']:.6f}" if calib['intercept'] is not None else "N/A"
            cv_error = f"{calib['cv_error']:.6f}" if calib['cv_error'] is not None else "N/A"
            r_squared = f"{calib['r_squared']:.6f}" if calib['r_squared'] is not None else "N/A"
            
            print(f"{det_idx:<4} {slope:<12} {intercept:<12} {cv_error:<10} {r_squared:<8}")
        
        print(f"\n✅ Example completed successfully!")
        print(f"   Use calibration_notebook.ipynb for full analysis and visualization")
        
    except Exception as e:
        print(f"❌ Calibration failed: {str(e)}")
        return


if __name__ == "__main__":
    main()
