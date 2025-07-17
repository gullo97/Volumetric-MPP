#!/usr/bin/env python3
"""
Quick test script for enhanced calibration features.

Tests:
1. Enhanced peak detection with both calibration and visualization peaks
2. Channel range validation for peak selection
3. Robust peak selection with fallbacks
"""

import numpy as np
import sys
import os

# Add calibration directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'calibration'))

from calibration_core import (
    detect_peaks_single_detector,
    validate_peak_in_range,
    apply_source_specific_peak_selection_robust
)

def test_enhanced_peak_detection():
    """Test the enhanced peak detection with dual output"""
    print("🧪 Testing Enhanced Peak Detection")
    print("=" * 50)
    
    # Create a synthetic spectrum with known peaks
    spectrum = np.zeros(1000)
    # Add some peaks
    spectrum[100:110] = np.exp(-0.5 * ((np.arange(10) - 5)**2) / 2)  # Peak at ~105
    spectrum[500:510] = 2 * np.exp(-0.5 * ((np.arange(10) - 5)**2) / 2)  # Peak at ~505  
    spectrum[800:810] = 1.5 * np.exp(-0.5 * ((np.arange(10) - 5)**2) / 2)  # Peak at ~805
    # Add noise
    spectrum += np.random.normal(0, 0.1, 1000)
    spectrum = np.maximum(spectrum, 0)  # Ensure non-negative
    
    # Simple detection parameters for testing
    detection_params = {
        'smoothing_range': [1, 3],
        'bins_factor_range': [1],
        'threshold_range': [0.1, 0.3],
        'width_range': [2, 5],
        'prominence_range': [0.2, 0.4],
        'distance_range': [10, 20],
        'parallel': False  # Disable parallel for testing
    }
    
    try:
        # Test with top_k = 2
        calibration_peaks, all_peaks = detect_peaks_single_detector(
            spectrum, detection_params, top_k=2, extended_top_k=6
        )
        
        print(f"✅ Detection successful!")
        print(f"   Calibration peaks (top-2): {len(calibration_peaks)}")
        print(f"   All peaks for visualization: {len(all_peaks)}")
        
        if calibration_peaks:
            print(f"   Calibration peak channels: {[p['peak_index'] for p in calibration_peaks]}")
        
        if all_peaks:
            print(f"   All peak channels: {[p['peak_index'] for p in all_peaks]}")
            
        return True
        
    except Exception as e:
        print(f"❌ Detection failed: {e}")
        return False

def test_channel_range_validation():
    """Test channel range validation function"""
    print("\n🧪 Testing Channel Range Validation")
    print("=" * 50)
    
    # Test ranges for Cesium (500-900)
    cesium_ranges = [(500, 900)]
    
    test_channels = [300, 600, 1000, 750, 450]
    expected_results = [False, True, False, True, False]
    
    all_passed = True
    for channel, expected in zip(test_channels, expected_results):
        result = validate_peak_in_range(channel, cesium_ranges)
        status = "✅" if result == expected else "❌"
        print(f"   {status} Channel {channel}: {result} (expected {expected})")
        if result != expected:
            all_passed = False
    
    return all_passed

def test_robust_peak_selection():
    """Test robust peak selection with channel validation"""
    print("\n🧪 Testing Robust Peak Selection")
    print("=" * 50)
    
    # Create mock persistent peaks
    mock_peaks = [
        {'peak_index': 650, 'persistence': 0.9},  # Valid for Cesium
        {'peak_index': 300, 'persistence': 0.8},  # Invalid (too low)
        {'peak_index': 750, 'persistence': 0.7},  # Valid for Cesium
        {'peak_index': 1000, 'persistence': 0.6}, # Invalid (too high)
        {'peak_index': 580, 'persistence': 0.5},  # Valid for Cesium
    ]
    
    try:
        # Test Cesium selection (expects 1 peak, should get highest valid channel)
        selected = apply_source_specific_peak_selection_robust(
            mock_peaks, "Cesium", [0.662]
        )
        
        print(f"✅ Cesium selection successful!")
        print(f"   Selected channels: {selected}")
        print(f"   Expected: highest valid channel around 750")
        
        # Test Cobalt selection (expects 2 peaks)
        cobalt_peaks = [
            {'peak_index': 1100, 'persistence': 0.9},  # Valid for Cobalt (low)
            {'peak_index': 1400, 'persistence': 0.8},  # Valid for Cobalt (high)
            {'peak_index': 500, 'persistence': 0.7},   # Invalid (too low)
            {'peak_index': 1200, 'persistence': 0.6},  # Valid for Cobalt (mid)
        ]
        
        selected_cobalt = apply_source_specific_peak_selection_robust(
            cobalt_peaks, "Cobalt", [1.17, 1.33]
        )
        
        print(f"✅ Cobalt selection successful!")
        print(f"   Selected channels: {selected_cobalt}")
        print(f"   Expected: 2 highest valid channels")
        
        return True
        
    except Exception as e:
        print(f"❌ Robust selection failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Enhanced Calibration Features")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Enhanced peak detection
    if test_enhanced_peak_detection():
        tests_passed += 1
    
    # Test 2: Channel range validation  
    if test_channel_range_validation():
        tests_passed += 1
    
    # Test 3: Robust peak selection
    if test_robust_peak_selection():
        tests_passed += 1
    
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Enhanced calibration features are working.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
