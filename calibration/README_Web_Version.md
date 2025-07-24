# Web Interactive Calibration Fixer

A web-based version of the Interactive Calibration Fixer that runs in your browser, replacing the tkinter GUI with a modern web interface.

## Features

- **Web-based Interface**: No need for tkinter - runs in any modern web browser
- **Interactive Peak Selection**: Click on spectrum plots to select calibration peaks
- **Real-time Visualization**: Uses Plotly for interactive, responsive plots
- **Automatic Peak Detection**: Shows suggested peaks using volumetric persistence
- **Cross-validation**: Calculates CV RMSE for calibration quality assessment
- **Excel Export**: Export complete calibration results with summary statistics

## Requirements

The web version requires these Python packages:
- Flask (web framework)
- pandas (data manipulation)
- numpy (numerical computing)
- matplotlib (plotting backend)
- plotly (interactive plots)
- scikit-learn (calibration algorithms)
- openpyxl (Excel export)

## Quick Start

### Option 1: Using the Launcher (Recommended)

1. Run the launcher script:
   ```bash
   python launch_web_calibration_fixer.py
   ```

2. The launcher will:
   - Check for required packages
   - Install missing packages automatically
   - Start the web server

3. Open your web browser and go to: `http://localhost:5000`

### Option 2: Manual Setup

1. Install requirements:
   ```bash
   pip install -r web_requirements.txt
   ```

2. Start the web application:
   ```bash
   python web_interactive_calibration_fixer.py
   ```

3. Open your browser to: `http://localhost:5000`

## Usage

### 1. Load Calibration Results
- Click "Load Calibration Results" and select your CSV file
- The system will load detector calibration data

### 2. Filter Failed Detectors
- Set your CV Error Threshold (default: 0.001)
- Click "Filter Failed Detectors" 
- The system will identify detectors needing manual correction

### 3. Select Peaks for Calibration
- Navigate through failed detectors using Previous/Next buttons
- View spectra for Sodium, Cesium, and Cobalt sources
- Click on peaks in the plots to select them for calibration
- Auto-detected peaks are shown as blue diamonds for reference
- Selected peaks appear as colored circles

### 4. Apply Calibration
- Ensure you have selected the correct number of peaks for each source:
  - Sodium: 1 peak (0.511 MeV)
  - Cesium: 1 peak (0.662 MeV) 
  - Cobalt: 2 peaks (1.173, 1.332 MeV)
- Click "Apply Calibration" to perform linear regression
- Results show slope, intercept, R², and CV RMSE

### 5. Export Results
- Click "Export Results to Excel" when done
- Downloads complete calibration results for all 128 detectors
- Includes summary statistics sheet

## Interface Overview

### Control Panel
- **File Upload**: Load calibration results CSV
- **Threshold Setting**: Adjust CV error threshold for filtering
- **Export Button**: Download complete results

### Progress Panel
- **Status Information**: Shows current detector and progress
- **Navigation**: Move between detectors, skip, or apply calibration

### Main Content
- **Left Panel**: Peak selection information and clear button
- **Right Panel**: Interactive spectrum plots for all three sources

### Plot Interactions
- **Click**: Select/deselect peaks
- **Zoom**: Use Plotly controls to zoom in/out
- **Pan**: Drag to pan across the spectrum
- **Hover**: See exact channel and intensity values

## Data Format

### Input CSV Format
The calibration results CSV should contain columns:
- `Detector`: Detector ID (0-127)
- `CV_RMSE_MeV`: Cross-validation RMSE in MeV
- `Num_Calibration_Points`: Number of calibration points used
- `Slope_MeV_per_channel`: Calibration slope
- `Intercept_MeV`: Calibration intercept
- `R2_Score`: R² goodness of fit

### Excel Data Source
The system expects Excel file at `../Data/Dati_luglio.xlsx` with sheets:
- `Sodio I`: Sodium spectra data
- `Cesio I`: Cesium spectra data  
- `Cobalto I`: Cobalt spectra data

## Configuration

### Source Configuration
Edit the `sources` dictionary in `web_interactive_calibration_fixer.py`:
```python
self.sources = {
    'Sodium': {
        'sheet_name': 'Sodio I',
        'expected_energies': [0.511],
        'color': '#FF6B6B'
    },
    # ... other sources
}
```

### Data File Path
Update `self.data_file` to point to your Excel data file:
```python
self.data_file = "../Data/Dati_luglio.xlsx"
```

### Channel Filtering
Adjust minimum channel for analysis:
```python
self.min_channel = 250  # Start analysis from channel 250
```

## Troubleshooting

### Installation Issues
- Make sure you have Python 3.7+ installed
- Try upgrading pip: `python -m pip install --upgrade pip`
- On macOS, you might need: `python3` instead of `python`

### Port Already in Use
If port 5000 is busy, edit the last line in `web_interactive_calibration_fixer.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

### Memory Issues
For large datasets, consider:
- Reducing the number of auto-detected peaks shown
- Processing detectors in smaller batches
- Increasing system memory allocation

### Browser Compatibility
Tested with:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Advantages over tkinter Version

1. **Cross-platform**: Works on any system with a web browser
2. **No GUI dependencies**: No need to install tkinter or other GUI libraries
3. **Modern interface**: Responsive, professional-looking web interface
4. **Interactive plots**: Plotly provides better zoom, pan, and hover functionality
5. **Easy deployment**: Can be accessed remotely if needed
6. **Better scaling**: Handles large datasets more efficiently

## Technical Details

### Architecture
- **Backend**: Flask web framework with RESTful API
- **Frontend**: HTML/CSS/JavaScript with Plotly.js
- **Data Processing**: pandas and numpy for data manipulation
- **Machine Learning**: scikit-learn for calibration algorithms
- **Visualization**: Plotly for interactive plots

### API Endpoints
- `POST /upload_calibration`: Upload calibration results CSV
- `POST /filter_failed`: Filter detectors by CV error threshold
- `GET /get_detector_data/<id>`: Get spectrum data for detector
- `POST /apply_calibration`: Apply calibration with selected peaks
- `GET /export_results`: Download Excel results file

### Performance
- Automatic peak detection runs in parallel when possible
- Client-side plot interactions for responsive UI
- Efficient data serialization between backend and frontend

## Support

For issues specific to the web version:
1. Check browser console for JavaScript errors
2. Verify all required Python packages are installed
3. Ensure data files are in the correct location
4. Check Flask server logs for backend errors
