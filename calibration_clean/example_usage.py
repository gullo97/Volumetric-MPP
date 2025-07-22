"""
Example usage of the clean calibration module.

This script demonstrates how to use the refactored calibration code
without the complex parameter optimization.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import the clean calibration module
from calibration_core import (
    process_source_data,
    calibrate_detector_array, 
    find_poor_detectors
)
from calibration_plots import (
    plot_calibration_quality_overview,
    plot_single_detector_calibration
)
from source_config import (
    get_source_config,
    print_source_info,
    GLOBAL_DETECTION_PARAMS
)


def main():
    """Main calibration workflow example."""
    
    print("🧪 Clean Calibration Module Example")
    print("=" * 50)
    
    # Show available sources
    print("\n📊 Available sources:")
    print_source_info()
    
    # Show global parameters being used
    print(f"\n🔧 Global parameters:")
    total_combinations = 1
    for key in ['smoothing_range', 'threshold_range', 'prominence_range']:
        if key in GLOBAL_DETECTION_PARAMS:
            total_combinations *= len(GLOBAL_DETECTION_PARAMS[key])
    print(f"   Total parameter combinations: {total_combinations}")
    
    # Example configuration
    DATA_FILE_PATH = "../DATI_Gullo1.xlsx/Dati_luglio.xlsx"
    SOURCE_MAPPING = {
        "Sodium": "Sodio I",
        "Cesium": "Cesio I", 
        "Cobalt": "Cobalto I"
    }
    
    print(f"\n📁 Processing data from: {DATA_FILE_PATH}")
    
    # Process sources
    sources_data = {}
    expected_energies = {}
    
    for source_name, sheet_name in SOURCE_MAPPING.items():
        print(f"\n📡 Processing {source_name}...")
        
        # Get source config
        source_config = get_source_config(source_name)
        source_config['source_name'] = source_name
        source_config['sheet_name'] = sheet_name
        expected_energies[source_name] = source_config['energies']
        
        try:
            # Process with global parameters
            spectra, peaks = process_source_data(
                DATA_FILE_PATH,
                source_config,
                GLOBAL_DETECTION_PARAMS
            )
            
            sources_data[source_name] = {
                'spectra': spectra,
                'peaks': peaks,
                'config': source_config
            }
            
            n_channels, n_detectors = spectra.shape
            print(f"   ✅ Success: {n_channels} channels, {n_detectors} detectors")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    if not sources_data:
        print("❌ No data processed successfully")
        return
    
    # Perform calibration
    print(f"\n🎯 Performing calibration...")
    calibration_results = calibrate_detector_array(sources_data, expected_energies)
    
    # Find poor detectors
    poor_detectors = find_poor_detectors(calibration_results, cv_threshold=0.1)
    
    # Show results
    successful = sum(1 for r in calibration_results.values() 
                    if r['calibration']['slope'] is not None)
    print(f"   Successful calibrations: {successful}/{len(calibration_results)}")
    print(f"   Poor detectors: {len(poor_detectors)}")
    
    # Generate plots
    print(f"\n📊 Generating plots...")
    
    # Quality overview
    plot_calibration_quality_overview(calibration_results)
    
    # Example single detector
    if 0 in calibration_results:
        plot_single_detector_calibration(0, calibration_results[0])
    
    print(f"\n✅ Example complete!")


if __name__ == "__main__":
    main()
