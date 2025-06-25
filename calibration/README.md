# Detector Calibration with Volumetric Persistence

This directory contains a complete, user-friendly calibration system for gamma-ray detectors using radioactive sources and the volumetric persistence peak detection method.

## 📁 Directory Structure

```
calibration/
├── calibration_notebook.ipynb    # Main interactive notebook
├── calibration_core.py           # Core calibration functions
├── calibration_plots.py          # Visualization functions
├── source_config.py              # Radioactive source configurations
└── README.md                     # This file
```

## ✨ Features

- **Multiple Data Formats**: Support for Excel (.xlsx) and CSV files
- **Predefined Sources**: Built-in configurations for common radioactive sources
- **Automated Peak Detection**: Uses volumetric persistence from the main repository
- **Custom Hyperparameters**: User-friendly configuration of detection parameters
- **Auto-Optimization**: Smart parameter selection based on source energy categories
- **Linear Calibration**: Energy calibration with cross-validation error estimation
- **Quality Assessment**: Automatic identification of poor-performing detectors
- **Comprehensive Visualization**: Multiple plot types for analysis and validation
- **Export Functionality**: Save calibration parameters and results

## 🚀 Quick Start

1. **Open the notebook**: Start with `calibration_notebook.ipynb`
2. **Configure your data**: Edit the configuration section (Section 2)
3. **Run all cells**: Execute the notebook step by step
4. **Review results**: Analyze plots and exported data

## 📊 Supported Radioactive Sources

The system includes predefined configurations for:

| Source | Energies (MeV) | Description |
|--------|----------------|-------------|
| **Sodium** | 0.511 | Na-22 positron annihilation peak |
| **Cobalt** | 1.17, 1.33 | Co-60 dual gamma rays |
| **Cesium** | 0.662 | Cs-137 single gamma ray |
| **Americium** | 0.0595 | Am-241 low energy gamma |
| **Barium** | 0.081, 0.356 | Ba-133 multiple peaks |
| **Europium** | 0.122, 0.244, 0.344, 0.779, 0.964, 1.408 | Eu-152 multi-peak calibration |
| **Manganese** | 0.835 | Mn-54 single gamma ray |

## ⚙️ Configuration

### Excel Files
```python
DATA_FILE_PATH = "path/to/your/data.xlsx"
FILE_FORMAT = "excel"

EXCEL_SHEET_MAPPING = {
    "Sodium": "Sheet_Sodium",
    "Cobalt": "Sheet_Cobalt", 
    "Cesium": "Sheet_Cesium"
}
```

### CSV Files
```python
DATA_FILE_PATH = "not_used_for_csv"
FILE_FORMAT = "csv"

CSV_FILE_MAPPING = {
    "Sodium": "path/to/sodium_data.csv",
    "Cobalt": "path/to/cobalt_data.csv",
    "Cesium": "path/to/cesium_data.csv"
}
```

## 📈 Workflow

1. **Data Loading**: Automatically loads spectra from configured sources
2. **Peak Detection**: Uses optimized volumetric persistence parameters per source
3. **Calibration**: Performs linear regression with cross-validation
4. **Quality Check**: Identifies detectors with poor calibration performance
5. **Visualization**: Generates comprehensive analysis plots
6. **Export**: Saves calibration parameters as CSV

## 🎯 Key Functions

### Core Functions (`calibration_core.py`)
- `process_source_data()`: Load and process spectral data
- `calibrate_detector_array()`: Perform calibration across all detectors
- `find_poor_detectors()`: Identify problematic detectors
- `channel_to_energy()`: Convert channels to energy using calibration

### Hyperparameter Functions (`calibration_core.py`)
- `create_parameter_grid()`: Create detection parameter grids
- `print_parameter_grid()`: Display parameter configurations
- `get_optimized_parameter_suggestions()`: Get energy-category suggestions
- `create_custom_detection_params()`: Build custom parameter sets

### Visualization (`calibration_plots.py`)
- `plot_calibration_quality_overview()`: Summary plots of calibration quality
- `plot_single_detector_calibration()`: Individual detector analysis
- `plot_combined_source_spectra()`: Multi-source spectrum overlay
- `plot_3d_energy_spectrum()`: 3D visualization in energy space
- `plot_2d_energy_heatmap()`: 2D energy heatmap

### Configuration (`source_config.py`)
- `get_source_config()`: Get complete source configuration
- `print_source_info()`: Display available sources
- `get_detection_params()`: Get optimized detection parameters

## 📊 Output Files

- **`detector_calibration_results.csv`**: Complete calibration parameters
  - Detector index, slope, intercept, CV error, R², quality flag
- **Various plots**: Automatically displayed in notebook
  - Calibration quality overview
  - Individual detector calibrations
  - Energy space visualizations
  - Source spectrum comparisons

## 🔧 Advanced Configuration

### Hyperparameter Customization

The system provides flexible hyperparameter configuration for volumetric persistence:

#### 1. Auto-Optimization (Recommended)
```python
USE_AUTO_OPTIMIZATION = True
```
Automatically selects optimal parameters based on source energy category.

#### 2. Custom Global Parameters
```python
USE_CUSTOM_GLOBAL = True

CUSTOM_DETECTION_PARAMS = create_parameter_grid(
    smoothing_config={'min': 1, 'max': 5, 'steps': 3},
    threshold_config={'min': 0.01, 'max': 0.1, 'steps': 4},
    prominence_config={'min': 0.1, 'max': 0.5, 'steps': 3}
)
```

#### 3. Source-Specific Parameters
```python
USE_SOURCE_SPECIFIC = True

SOURCE_SPECIFIC_PARAMS = {
    "Cesium": create_parameter_grid(
        threshold_config={'min': 0.01, 'max': 0.05, 'steps': 3}
    ),
    "Cobalt": create_parameter_grid(
        threshold_config={'min': 0.05, 'max': 0.2, 'steps': 3}
    )
}
```

#### Parameter Meanings:
- **smoothing_range**: Spectrum smoothing window sizes (1-10)
- **bins_factor_range**: Channel aggregation factors (1-3)
- **threshold_range**: Peak detection thresholds (0.01-0.5)
- **width_range**: Expected peak widths in channels (1-10)
- **prominence_range**: Peak prominence requirements (0.05-1.0)
- **distance_range**: Minimum peak separation (3-30)

### Detection Parameters
The system automatically selects optimized parameters based on source energy:
- **Low Energy** (e.g., Am-241): Fine resolution, low thresholds
- **Medium Energy** (e.g., Cs-137, Na-22): Balanced parameters
- **High Energy** (e.g., Co-60): Coarser resolution, higher thresholds
- **Multi-Peak** (e.g., Eu-152): Extended parameter ranges

### Quality Thresholds
```python
CALIBRATION_SETTINGS = {
    "cv_error_threshold": 0.05,  # MeV
    "count_threshold": 100,      # For 3D plots
    "use_optimized_params": True,
    "parallel_processing": True,
}
```

## 🚨 Troubleshooting

### Common Issues

1. **"No peaks detected"**
   - Check if your data has sufficient counts
   - Verify energy ranges match your source energies
   - Try adjusting detection parameters

2. **"Poor calibration quality"**
   - Ensure peaks are correctly identified
   - Check for detector hardware issues
   - Verify source energy assignments

3. **"File not found"**
   - Check file paths in configuration
   - Ensure sheet names match exactly (for Excel)
   - Verify file permissions

### Performance Tips

- Enable parallel processing for faster peak detection
- Use appropriate top_k values (2-5 for most sources)
- Filter out detectors with known hardware issues

## 🔄 Compared to Old Implementation

This new system improves upon the old calibration notebook by:

- **Modular Design**: Separated functionality into focused modules
- **User-Friendly**: Clear configuration and guided workflow
- **Flexible Data Input**: Support for multiple file formats
- **Better Error Handling**: Robust error checking and reporting
- **Enhanced Visualization**: More comprehensive and publication-ready plots
- **Automated Configuration**: Smart parameter selection per source type
- **Export Functionality**: Structured data export for downstream use

## 📚 Dependencies

This calibration system requires:
- Core volumetric persistence functions (`../core.py`)
- Utility functions (`../utils.py`)
- Standard scientific Python libraries (numpy, pandas, matplotlib, scipy)

## 💡 Tips for Best Results

1. **Source Selection**: Use multiple sources spanning your energy range of interest
2. **Peak Quality**: Ensure detected peaks are clearly visible and well-separated
3. **Detector Coverage**: Calibrate as many detectors as possible for consistency
4. **Regular Validation**: Repeat calibration periodically to check stability
5. **Documentation**: Keep records of calibration parameters and quality metrics

## 🤝 Contributing

To extend this calibration system:
1. Add new sources to `source_config.py`
2. Implement custom detection parameters for special cases
3. Add new visualization functions to `calibration_plots.py`
4. Enhance the core calibration algorithms in `calibration_core.py`

---

**Happy Calibrating! 🎯**
