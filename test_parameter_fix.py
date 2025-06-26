#!/usr/bin/env python3
"""
Test script to verify that the parameter range generation fix works correctly.
"""

import sys
import os
sys.path.append('calibration')

from calibration_core import create_parameter_grid, print_parameter_grid

def test_parameter_types():
    """Test that parameter ranges have correct types."""
    print("🧪 Testing parameter type generation...")
    print("=" * 50)
    
    # Test with parameters that should be integers
    params = create_parameter_grid(
        smoothing_config={'min': 1, 'max': 5.7, 'steps': 4},      # Should become integers
        bins_factor_config={'min': 1, 'max': 3, 'steps': 3},      # Should be integers
        threshold_config={'min': 0.01, 'max': 0.1, 'steps': 4},   # Can be floats
        width_config={'min': 1.5, 'max': 6.8, 'steps': 4},       # Should be rounded floats
        prominence_config={'min': 0.1, 'max': 0.5, 'steps': 3},   # Can be floats
        distance_config={'min': 5.2, 'max': 15.7, 'steps': 4}    # Should become integers
    )
    
    print("Generated parameters:")
    print_parameter_grid(params)
    
    print("\n🔍 Type checking:")
    
    # Check smoothing_range (should be integers)
    smoothing_values = params['smoothing_range']
    print(f"Smoothing range: {smoothing_values}")
    print(f"  Types: {[type(v).__name__ for v in smoothing_values]}")
    assert all(isinstance(v, int) for v in smoothing_values), "Smoothing values should be integers"
    print("  ✅ All smoothing values are integers")
    
    # Check bins_factor_range (should be integers)
    bins_values = params['bins_factor_range']
    print(f"Bins factor range: {bins_values}")
    print(f"  Types: {[type(v).__name__ for v in bins_values]}")
    assert all(isinstance(v, int) for v in bins_values), "Bins factor values should be integers"
    print("  ✅ All bins factor values are integers")
    
    # Check distance_range (should be integers)
    distance_values = params['distance_range']
    print(f"Distance range: {distance_values}")
    print(f"  Types: {[type(v).__name__ for v in distance_values]}")
    assert all(isinstance(v, int) for v in distance_values), "Distance values should be integers"
    print("  ✅ All distance values are integers")
    
    # Check width_range (should be floats, reasonably rounded)
    width_values = params['width_range']
    print(f"Width range: {width_values}")
    print(f"  Types: {[type(v).__name__ for v in width_values]}")
    # Check that values are reasonably rounded (1 decimal place)
    for v in width_values:
        assert abs(v - round(v, 1)) < 1e-10, f"Width value {v} should be rounded to 1 decimal place"
    print("  ✅ All width values are properly rounded")
    
    # Check threshold and prominence ranges (can be floats)
    threshold_values = params['threshold_range']
    prominence_values = params['prominence_range']
    print(f"Threshold range: {threshold_values}")
    print(f"Prominence range: {prominence_values}")
    print("  ✅ Threshold and prominence can be any numeric type")
    
    print("\n🎉 All parameter type tests passed!")
    
    # Test edge cases
    print("\n🧪 Testing edge cases...")
    
    # Single step
    single_params = create_parameter_grid(
        smoothing_config={'min': 3, 'max': 3, 'steps': 1},
        distance_config={'min': 10.7, 'max': 10.7, 'steps': 1}
    )
    
    print(f"Single step smoothing: {single_params['smoothing_range']}")
    print(f"Single step distance: {single_params['distance_range']}")
    assert single_params['smoothing_range'] == [3]
    assert single_params['distance_range'] == [10]  # Should be rounded to int
    print("  ✅ Single step cases work correctly")
    
    # Two steps
    two_params = create_parameter_grid(
        smoothing_config={'min': 1, 'max': 5, 'steps': 2},
        distance_config={'min': 5.1, 'max': 15.9, 'steps': 2}
    )
    
    print(f"Two step smoothing: {two_params['smoothing_range']}")
    print(f"Two step distance: {two_params['distance_range']}")
    assert two_params['smoothing_range'] == [1, 5]
    assert two_params['distance_range'] == [5, 15]  # Should be rounded to int
    print("  ✅ Two step cases work correctly")
    
    print("\n✅ All tests passed! Parameter generation is working correctly.")

if __name__ == "__main__":
    test_parameter_types()
