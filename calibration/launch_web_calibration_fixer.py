#!/usr/bin/env python3
"""
Launch script for Web Interactive Calibration Fixer

This script will:
1. Check if required packages are installed
2. Install them if needed
3. Launch the web application

Usage:
    python launch_web_calibration_fixer.py
"""

import sys
import subprocess
import os

def check_and_install_requirements():
    """Check if required packages are installed and install if needed."""
    requirements_file = os.path.join(os.path.dirname(__file__), 'web_requirements.txt')
    
    if not os.path.exists(requirements_file):
        print("Error: web_requirements.txt not found!")
        return False
    
    print("Checking required packages...")
    
    try:
        # Try importing key packages
        import flask
        import pandas
        import numpy
        import plotly
        import sklearn
        print("✓ All required packages are already installed")
        return True
    except ImportError as e:
        print(f"Missing package detected: {e}")
        print("Installing required packages...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", requirements_file
            ])
            print("✓ Successfully installed all required packages")
            return True
        except subprocess.CalledProcessError:
            print("✗ Failed to install required packages")
            print("Please try installing manually with:")
            print(f"pip install -r {requirements_file}")
            return False

def launch_application():
    """Launch the web application."""
    script_path = os.path.join(os.path.dirname(__file__), 'web_interactive_calibration_fixer.py')
    
    if not os.path.exists(script_path):
        print("Error: web_interactive_calibration_fixer.py not found!")
        return False
    
    print("\n" + "="*60)
    print("Starting Web Interactive Calibration Fixer...")
    print("="*60)
    print("Once the server starts, open your web browser and go to:")
    print("    http://localhost:5001")
    print("="*60)
    print("Press Ctrl+C to stop the server when done.")
    print("="*60 + "\n")
    
    try:
        subprocess.run([sys.executable, script_path])
        return True
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
        return True
    except Exception as e:
        print(f"Error launching application: {e}")
        return False

def main():
    """Main function."""
    print("Web Interactive Calibration Fixer Launcher")
    print("==========================================\n")
    
    # Check and install requirements
    if not check_and_install_requirements():
        sys.exit(1)
    
    # Launch the application
    if not launch_application():
        sys.exit(1)
    
    print("Application closed successfully.")

if __name__ == "__main__":
    main()
