# Clean Calibration Module

This is a refactored, streamlined version of the detector calibration code that removes all the complex parameter optimization functionality that wasn't working properly and focuses on what actually works.

## What's Included

### Working Files:
- **`calibration_notebook.ipynb`** - Clean, maintainable notebook with only working functionality
- **`calibration_core.py`** - Core calibration functions with global parameters
- **`calibration_plots.py`** - All plotting and visualization functions
- **`source_config.py`** - Source configurations with simple global parameters

### Key Improvements:
1. **Removed broken parameter optimization** - All the complex parameter grid creation and optimization code that wasn't working
2. **Single global parameter set** - Uses one well-tested parameter configuration for all sources
3. **Simplified peak detection** - Straightforward volumetric persistence with physics-based selection
4. **Maintained all plotting capabilities** - All visualization functions preserved
5. **Clean notebook structure** - Only cells that actually produce results

## What Was Removed

- All hyperparameter optimization cells (sections 2.5, parameter exploration, etc.)
- Complex parameter grid creation functions
- Source-specific parameter optimization
- Energy-based parameter selection
- All the "examples" and "exploration" code that cluttered the notebook

## Global Parameters

The system now uses a single, tested set of global detection parameters defined in `source_config.py`:

```python
GLOBAL_DETECTION_PARAMS = {
    "smoothing_range": [1, 3, 5],
    "bins_factor_range": [1, 2],
    "threshold_range": [0.01, 0.05, 0.1],
    "width_range": [1.0, 3.0, 5.0],
    "prominence_range": [0.1, 0.3, 0.5],
    "distance_range": [5, 10, 15],
    "merging_range": 5,
    "tol": 1,
    "parallel": True
}
```

This gives you 3×2×3×3×3×3 = 486 parameter combinations - enough for good detection without the complexity.

## Usage

Simply run the clean notebook from start to finish. The workflow is:

1. **Setup** - Import modules and check configuration
2. **Configuration** - Set your data paths and sources
3. **Data Processing** - Process all sources with global parameters
4. **Calibration** - Perform linear calibration
5. **Visualization** - Generate all analysis plots:
   - Calibration quality overview
   - Individual detector analysis
   - Source spectrum visualization
   - 3D energy space plots
   - 2D heatmaps
   - Poor detector investigation
6. **Export** - Save results to CSV

## Benefits

- **Maintainable**: Clear, simple code structure
- **Reliable**: Uses only tested, working functionality
- **Complete**: All visualization and analysis capabilities preserved
- **Fast**: No complex optimization loops
- **Understandable**: Clear workflow without unnecessary complexity

## Migration from Old Code

If you have existing analysis using the old calibration folder, you can:

1. Update import statements to use the new module
2. Remove any parameter optimization code from your scripts
3. Use the global parameters instead of custom parameter grids
4. The rest of your analysis code should work unchanged

The API for the core functions (`process_source_data`, `calibrate_detector_array`, etc.) remains the same.
