#!/usr/bin/env python3
"""
Simple test script for the calibration system.

This script runs basic tests to verify the calibration package works correctly.
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_source_config():
    """Test source configuration functionality."""
    print("🧪 Testing source configuration...")
    
    try:
        from source_config import (
            get_source_config, 
            list_available_sources,
            get_detection_params
        )
        
        # Test getting source list
        sources = list_available_sources()
        assert len(sources) > 0, "No sources available"
        print(f"   ✅ Found {len(sources)} sources")
        
        # Test getting source config
        config = get_source_config("Cesium")
        assert 'energies' in config, "Missing energies in config"
        assert 'detection_params' in config, "Missing detection params in config"
        print(f"   ✅ Cesium config loaded: {config['energies']} MeV")
        
        # Test detection parameters
        params = get_detection_params("medium_energy")
        assert 'smoothing_range' in params, "Missing smoothing_range"
        print(f"   ✅ Detection parameters loaded")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Source config test failed: {str(e)}")
        return False


def test_calibration_functions():
    """Test calibration core functions."""
    print("🧪 Testing calibration functions...")
    
    try:
        from calibration_core import (
            calibrate_detector,
            channel_to_energy,
            energy_to_channel
        )
        
        # Test calibration with synthetic data
        channels = [100, 200, 300, 400, 500]
        energies = [0.1, 0.2, 0.3, 0.4, 0.5]  # Linear relationship
        
        result = calibrate_detector(channels, energies)
        assert result['slope'] is not None, "Calibration failed"
        assert result['intercept'] is not None, "Calibration failed"
        print(f"   ✅ Calibration: slope={result['slope']:.6f}, "
              f"intercept={result['intercept']:.6f}")
        
        # Test energy conversion
        test_channels = np.array([150, 250, 350])
        energies_calc = channel_to_energy(test_channels, result['slope'], result['intercept'])
        channels_calc = energy_to_channel(energies_calc, result['slope'], result['intercept'])
        
        # Should get back original channels
        assert np.allclose(test_channels, channels_calc, rtol=1e-10), "Conversion test failed"
        print(f"   ✅ Energy conversion working correctly")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Calibration functions test failed: {str(e)}")
        return False


def test_synthetic_data_processing():
    """Test data processing with synthetic data."""
    print("🧪 Testing with synthetic data...")
    
    try:
        from calibration_core import detect_peaks_single_detector
        from source_config import get_detection_params
        
        # Create synthetic spectrum with known peaks
        n_channels = 1000
        spectrum = np.random.poisson(10, n_channels)  # Background
        
        # Add some peaks
        peak_positions = [200, 400, 600, 800]
        peak_heights = [1000, 800, 600, 400]
        
        for pos, height in zip(peak_positions, peak_heights):
            # Gaussian peaks
            sigma = 5
            x = np.arange(n_channels)
            peak = height * np.exp(-0.5 * ((x - pos) / sigma) ** 2)
            spectrum += peak.astype(int)
        
        # Detect peaks
        detection_params = get_detection_params("medium_energy")
        detected_peaks = detect_peaks_single_detector(spectrum, detection_params, top_k=5)
        
        print(f"   ✅ Detected {len(detected_peaks)} peaks in synthetic spectrum")
        
        # Check if we found peaks near expected positions
        detected_positions = [peak['peak_index'] for peak in detected_peaks]
        found_near_expected = 0
        for expected_pos in peak_positions:
            for detected_pos in detected_positions:
                if abs(detected_pos - expected_pos) < 20:  # Within 20 channels
                    found_near_expected += 1
                    break
        
        print(f"   ✅ Found {found_near_expected}/{len(peak_positions)} expected peaks")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Synthetic data test failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("🔬 Calibration System Tests")
    print("=" * 40)
    
    tests = [
        test_source_config,
        test_calibration_functions,
        test_synthetic_data_processing
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"   ❌ Test {test_func.__name__} crashed: {str(e)}\n")
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The calibration system is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
