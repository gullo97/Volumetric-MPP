# Enhanced Volumetric MPP Calibration - Implementation Summary

## Overview
This document summarizes the improvements made to the volumetric persistence calibration system to address two key issues:

1. **Limited barcode visualization** - Only top-K peaks were shown, hiding persistence information
2. **Insufficient peak selection robustness** - Top-K ranking sometimes missed desired peaks due to variations across detectors

## Issue 1: Enhanced Barcode Visualization

### Problem
- Previously, only the top-K most persistent peaks were stored and displayed in barcode plots
- This prevented users from seeing how well the parameter grid separated true peaks from noise
- No visibility into the persistence landscape beyond the selected peaks

### Solution
- **Modified `detect_peaks_single_detector()`** to return both calibration peaks (top-K) and extended peaks (typically 3×K or minimum 20)
- **Updated peak storage structure** to include both `calibration_peaks` and `all_peaks` 
- **Enhanced plotting functions** to use extended peaks for barcode visualization while maintaining top-K for calibration

### Implementation Details
```python
# New function signature
def detect_peaks_single_detector(spectrum, detection_params, top_k=5, extended_top_k=None):
    # Returns: (calibration_peaks, all_peaks_for_viz)
    
# New peak storage structure  
detector_peaks = {
    'calibration_peaks': [...],  # Top-K for calibration
    'all_peaks': [...]           # Extended peaks for visualization
}
```

### Benefits
- Users can now see the full persistence landscape in barcode plots
- Better assessment of parameter grid effectiveness at separating signal from noise
- Improved debugging capabilities for peak detection issues

## Issue 2: Robust Peak Selection with Channel Range Validation

### Problem
- Top-K persistence ranking could miss desired peaks due to:
  - Slight ranking variations across detectors
  - Noise-induced peaks occasionally outranking true peaks
  - No physics-based validation of peak positions

### Solution
- **Added source-specific channel range validation** before physics-based peak selection
- **Implemented iterative peak selection** that validates against expected channel ranges
- **Enhanced fallback mechanisms** for robustness

### Implementation Details

#### 1. Channel Range Configuration (`source_config.py`)
```python
SOURCE_CHANNEL_RANGES = {
    "Sodium": {
        "expected_ranges": [(400, 700)],  # ~511 keV
        "description": "511 keV annihilation peak"
    },
    "Cesium": {
        "expected_ranges": [(500, 900)],  # ~662 keV  
        "description": "662 keV gamma peak"
    },
    "Cobalt": {
        "expected_ranges": [(900, 1400), (1200, 1700)],  # ~1173, 1332 keV
        "description": "1173 and 1332 keV gamma peaks"
    },
    # ... other sources
}
```

#### 2. Enhanced Peak Selection Process
```python
def apply_source_specific_peak_selection_robust(persistent_peaks, source_name, expected_energies):
    # Step 1: Filter peaks by expected channel ranges
    valid_peaks = [peak for peak in persistent_peaks 
                   if validate_peak_in_range(peak['peak_index'], expected_ranges)]
    
    # Step 2: Apply source-specific selection within valid peaks
    # (same physics-based rules as before, but only on validated peaks)
```

#### 3. Three-Step Selection Process
1. **Volumetric Persistence Filtering**: Get top-K most persistent peaks
2. **Channel Range Validation**: Filter peaks that fall within expected ranges
3. **Physics-Based Selection**: Apply source-specific rules (highest channel, etc.)

### Benefits
- More robust calibration that handles ranking variations between detectors
- Physics-informed peak selection reduces false positives
- Graceful fallback to original method if no peaks found in expected ranges
- Better handling of noisy spectra or unusual detector responses

## Performance Optimizations

### Reduced Parameter Grid (for faster testing)
```python
# Before: ~3000+ parameter combinations
CUSTOM_DETECTION_PARAMS = create_parameter_grid(
    smoothing_config={'min': 1, 'max': 5, 'steps': 3},      
    threshold_config={'min': 0.01, 'max': 0.2, 'steps': 5},    
    prominence_config={'min': 0.01, 'max': 1.0, 'steps': 8},
    # ... other parameters
)

# After: 216 parameter combinations (2×2×3×3×3×2)
CUSTOM_DETECTION_PARAMS = create_parameter_grid(
    smoothing_config={'min': 1, 'max': 3, 'steps': 2},      
    threshold_config={'min': 0.01, 'max': 0.1, 'steps': 3},    
    prominence_config={'min': 0.1, 'max': 0.5, 'steps': 3},
    # ... reduced parameters
)
```

## Files Modified

### Core Files
- `calibration/calibration_core.py`:
  - Enhanced `detect_peaks_single_detector()` with dual output
  - Added `validate_peak_in_range()` function
  - Added `apply_source_specific_peak_selection_robust()` function
  - Updated `detect_peaks_multi_detector()` to handle new structure
  - Modified `calibrate_detector_array()` to use robust selection

- `calibration/source_config.py`:
  - Added `SOURCE_CHANNEL_RANGES` with expected channel ranges for each source
  - Enhanced source configuration with validation ranges

- `calibration/calibration_plots.py`:
  - Updated `plot_combined_source_spectra()` to handle new peak structure
  - Enhanced barcode plotting to show extended peaks
  - Updated 3D and 2D visualization functions for new structure

### Notebook Updates
- `calibration/calibration_notebook.ipynb`:
  - Updated documentation with new features
  - Reduced parameter grid for faster testing
  - Modified peak counting to handle new structure
  - Enhanced visualization calls

## Usage Examples

### Basic Usage (No Changes Required)
The calibration pipeline works exactly as before for basic usage:
```python
# Standard calibration workflow unchanged
sources_data = {}
calibration_results = calibrate_detector_array(sources_data, expected_energies)
```

### Advanced Usage - Custom Channel Ranges
```python
# Modify channel ranges for your detector setup
SOURCE_CHANNEL_RANGES["Cesium"]["expected_ranges"] = [(450, 850)]  # Adjust for your detector

# Enable extended barcode visualization
plot_combined_source_spectra(
    detector_idx, 
    sources_data,
    show_all_peaks_in_barcode=True  # Shows extended peaks
)
```

### Testing the Enhancements
```python
# Test channel validation
from calibration_core import validate_peak_in_range
result = validate_peak_in_range(650, [(500, 900)])  # True for Cesium range

# Test robust selection
selected_peaks = apply_source_specific_peak_selection_robust(
    persistent_peaks, "Cesium", [0.662]
)
```

## Expected Improvements

### Calibration Quality
- Better peak selection consistency across detectors
- Reduced impact of noise-induced false peaks
- More physically meaningful peak assignments

### Visualization and Debugging
- Complete persistence landscape visible in barcode plots
- Better understanding of parameter grid effectiveness
- Improved troubleshooting capabilities for problematic detectors

### Robustness
- Graceful handling of ranking variations
- Fallback mechanisms for edge cases
- Physics-informed validation reduces systematic errors

## Recommendations for Usage

1. **First Run**: Use the default settings to test the enhanced features
2. **Channel Range Tuning**: Adjust `SOURCE_CHANNEL_RANGES` based on your detector's actual calibration
3. **Parameter Optimization**: Start with the reduced parameter grid, then expand if needed
4. **Validation**: Check barcode plots to ensure good separation between signal and noise
5. **Troubleshooting**: Use extended peak visualization to diagnose detection issues

## Future Enhancements

Potential areas for further improvement:
- Adaptive channel range estimation based on rough calibration
- Machine learning-based peak classification
- Automated parameter grid optimization
- Real-time calibration monitoring and drift detection

---

## Testing

A test script `test_enhanced_calibration.py` is provided to verify the core functionality:
```bash
cd "/Users/gullo/Documents/PHD/Volumetric MPP"
python test_enhanced_calibration.py
```

This implementation maintains backward compatibility while adding powerful new features for more robust and informative detector calibration.
