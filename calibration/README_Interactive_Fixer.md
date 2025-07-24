# Interactive Detector Calibration Fixer

This tool provides a user-friendly GUI interface for manually correcting failed detector calibrations by interactively selecting peaks from gamma spectroscopy data.

## Overview

The Interactive Calibration Fixer is designed to handle cases where the automatic calibration fails due to:
- Heavy noise in detector signals
- Unexpected peak positions
- Complex peak structures
- Low signal-to-noise ratios

## Features

- **Interactive Peak Selection**: Click directly on spectrum plots to select calibration peaks
- **Multi-Source Analysis**: Displays spectra from Sodium, Cesium, and Cobalt sources simultaneously
- **Automatic Peak Suggestions**: Shows automatically detected peaks as guidance
- **Real-time Feedback**: Immediate visual feedback on selected peaks and calibration status
- **Quality Metrics**: Calculates R², CV error, and other calibration quality metrics
- **Excel Export**: Comprehensive export of all 128 detector calibrations to Excel format

## Requirements

- Python 3.7+
- Required packages:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn
  - tkinter (usually included with Python)
  - openpyxl (for Excel export)

## Quick Start

### 1. Run the Detector Calibration Notebook First

Before using the interactive fixer, you need to run the main calibration notebook:

```bash
# Open the calibration notebook and run all cells
jupyter notebook detector_calibration.ipynb
```

This will generate:
- `detector_calibration_results.csv` - Summary of all calibration results
- `detector_calibration_functions.json` - Detailed calibration parameters

### 2. Launch the Interactive Fixer

```bash
python launch_calibration_fixer.py
```

Or directly:

```bash
python interactive_calibration_fixer_clean.py
```

### 3. Using the Interface

1. **Load Results**: Click "Load Calibration Results" and select your CSV file
2. **Set Threshold**: Adjust the CV Error Threshold (default: 0.001) to filter detectors
3. **Filter Detectors**: Click "Filter Failed Detectors" to identify problematic calibrations
4. **Select Peaks**: Click on peaks in the three spectrum plots to select calibration points
5. **Apply Calibration**: When you have the correct peaks selected, click "Apply Calibration"
6. **Navigate**: Use Previous/Next/Skip buttons to move through detectors
7. **Export**: When finished, export complete results to Excel

## Interface Guide

### Main Components

- **Control Panel**: Load data, set thresholds, export results
- **Progress Panel**: Shows current detector and navigation controls
- **Selected Peaks Panel**: Displays currently selected peaks and their information
- **Spectrum Plots**: Three interactive plots for Sodium, Cesium, and Cobalt sources

### Peak Selection Rules

- **Sodium**: Select 1 peak (0.511 MeV)
- **Cesium**: Select 1 peak (0.662 MeV) 
- **Cobalt**: Select 2 peaks (1.173 MeV and 1.332 MeV)

### Visual Indicators

- **Light blue dashed lines**: Automatically detected peak suggestions
- **Colored solid lines**: Your selected peaks (red for Sodium, teal for Cesium, blue for Cobalt)
- **Black dots**: Selected peak positions on the spectrum

## Tips for Effective Use

### Peak Selection Strategy

1. **Use Auto-Suggestions**: The light blue dashed lines show automatically detected peaks - these are often good starting points
2. **Look for Highest Peaks**: For each source, select the most prominent peaks in the expected energy ranges
3. **Cobalt Special Case**: For Cobalt, select the two highest-energy peaks (rightmost in the spectrum)
4. **Remove Wrong Selections**: Click near an existing selection to remove it

### Quality Assessment

- **R² Score**: Should be > 0.95 for good calibrations
- **CV RMSE**: Should be < 0.01 MeV for high-quality calibrations
- **Visual Check**: The calibration line should pass close to all selected points

### Troubleshooting

- **No Peaks Visible**: Try adjusting the plot zoom or check if the data file path is correct
- **Can't Select Peaks**: Make sure you're clicking directly on the spectrum lines
- **Poor Calibration Quality**: Try selecting more prominent peaks or different peak combinations

## Output Files

### Excel Export Structure

The exported Excel file contains two sheets:

1. **Complete_Calibration**: 
   - All 128 detectors with calibration parameters
   - Status indicators (Good/Acceptable/Failed)
   - Quality metrics for each detector

2. **Summary_Statistics**:
   - Overall calibration success rate
   - Mean and standard deviation of calibration parameters
   - Quality metric summaries

### Calibration Status Definitions

- **Good**: R² ≥ 0.95 and CV RMSE ≤ 0.01 MeV
- **Acceptable**: Has valid calibration but doesn't meet "Good" criteria
- **Failed**: No valid calibration could be determined

## Integration with Main Workflow

The interactive fixer is designed to complement the main calibration notebook:

1. Run the main calibration to get automatic results for most detectors
2. Use the interactive fixer to manually correct the remaining problematic cases
3. Export the final complete calibration dataset for use in analysis

## Technical Details

### Calibration Method

- **Linear Regression**: Energy = slope × Channel + intercept
- **Cross-Validation**: Leave-one-out CV for error estimation
- **Quality Metrics**: R², RMSE, and visual inspection

### Data Processing

- **Channel Filtering**: Removes channels < 250 to eliminate backscattering
- **Normalization**: Spectra normalized by area under curve
- **Peak Detection**: Uses volumetric persistence for automatic suggestions

## Support

If you encounter issues:

1. Check that all required Python packages are installed
2. Verify that the data files are in the correct location
3. Ensure the calibration notebook has been run first
4. Check the console output for error messages

For additional help, refer to the main calibration notebook documentation.
