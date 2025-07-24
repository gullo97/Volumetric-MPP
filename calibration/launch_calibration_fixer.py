#!/usr/bin/env python3
"""
Launch script for the Interactive Calibration Fixer

This script provides a simple way to launch the interactive calibration fixer
for manual correction of failed detector calibrations.

Usage:
    python launch_calibration_fixer.py
"""

import os
import sys
import subprocess

def main():
    """Launch the interactive calibration fixer."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fixer_script = os.path.join(script_dir, "interactive_calibration_fixer_clean.py")
    
    if not os.path.exists(fixer_script):
        print(f"Error: Could not find {fixer_script}")
        print("Make sure the interactive_calibration_fixer_clean.py file exists.")
        return 1
    
    print("Launching Interactive Calibration Fixer...")
    print("This will open a GUI window for manual calibration correction.")
    print("")
    print("Instructions:")
    print("1. Click 'Load Calibration Results' to load your CSV file")
    print("2. Set the CV Error Threshold and click 'Filter Failed Detectors'")
    print("3. Click on peaks in the spectra to select them for calibration")
    print("4. Use 'Apply Calibration' when you have selected the correct peaks")
    print("5. Navigate through detectors using Previous/Next/Skip buttons")
    print("6. Export final results to Excel when complete")
    print("")
    
    try:
        # Launch the GUI application
        subprocess.run([sys.executable, fixer_script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error launching calibration fixer: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nApplication closed by user.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
