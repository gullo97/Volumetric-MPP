#!/usr/bin/env python3
"""
Hyperparameter Configuration Examples

This script demonstrates the new hyperparameter configuration capabilities
of the calibration system.
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration_core import (
    create_parameter_grid,
    print_parameter_grid,
    get_optimized_parameter_suggestions,
    create_custom_detection_params
)
from source_config import get_source_config


def main():
    """Demonstrate hyperparameter configuration options."""
    
    print("🔧 Hyperparameter Configuration Examples")
    print("=" * 50)
    
    # Example 1: Basic parameter grid creation
    print("\n1️⃣ Basic Parameter Grid Creation:")
    print("-" * 35)
    
    basic_params = create_parameter_grid(
        smoothing_config={'min': 1, 'max': 5, 'steps': 3},
        threshold_config={'min': 0.01, 'max': 0.1, 'steps': 4}
    )
    print_parameter_grid(basic_params)
    
    # Example 2: Energy category optimization
    print("\n2️⃣ Energy Category Optimizations:")
    print("-" * 35)
    
    categories = ['low_energy', 'medium_energy', 'high_energy']
    for category in categories:
        print(f"\n{category.replace('_', ' ').title()}:")
        suggestions = get_optimized_parameter_suggestions(category)
        
        # Show key parameters only for brevity
        key_params = ['smoothing', 'threshold', 'prominence']
        for param in key_params:
            config = suggestions[param]
            print(f"  {param}: {config['min']}-{config['max']} "
                  f"({config['steps']} steps)")
    
    # Example 3: Source-based parameter creation
    print("\n3️⃣ Source-Based Parameter Creation:")
    print("-" * 35)
    
    sources = ['Cesium', 'Cobalt', 'Sodium']
    for source in sources:
        try:
            config = get_source_config(source)
            print(f"\n{source} ({config['energy_category']}):")
            
            params = create_custom_detection_params(source_name=source)
            
            # Show parameter ranges
            for param_name in ['smoothing_range', 'threshold_range', 'prominence_range']:
                if param_name in params:
                    values = params[param_name]
                    print(f"  {param_name}: {values}")
                    
        except Exception as e:
            print(f"  Error with {source}: {e}")
    
    # Example 4: Custom parameter modification
    print("\n4️⃣ Custom Parameter Modification:")
    print("-" * 35)
    
    # Start with Cesium optimization but modify specific parameters
    custom_params = create_custom_detection_params(
        source_name="Cesium",
        threshold_config={'min': 0.005, 'max': 0.08, 'steps': 6},  # Custom threshold
        smoothing_config={'min': 1, 'max': 8, 'steps': 4}          # Custom smoothing
    )
    
    print("\nCesium base + custom threshold & smoothing:")
    print_parameter_grid(custom_params)
    
    # Example 5: Performance comparison
    print("\n5️⃣ Performance Impact:")
    print("-" * 35)
    
    # Fast configuration (fewer combinations)
    fast_params = create_parameter_grid(
        smoothing_config={'min': 1, 'max': 3, 'steps': 2},
        threshold_config={'min': 0.05, 'max': 0.1, 'steps': 2},
        prominence_config={'min': 0.2, 'max': 0.4, 'steps': 2}
    )
    
    # Thorough configuration (many combinations)
    thorough_params = create_parameter_grid(
        smoothing_config={'min': 1, 'max': 8, 'steps': 5},
        threshold_config={'min': 0.01, 'max': 0.2, 'steps': 8},
        prominence_config={'min': 0.05, 'max': 0.8, 'steps': 6}
    )
    
    print("\nFast Configuration (fewer combinations):")
    print_parameter_grid(fast_params)
    
    print("\nThorough Configuration (many combinations):")
    print_parameter_grid(thorough_params)
    
    # Calculate combinations
    def count_combinations(params):
        total = 1
        for key in ['smoothing_range', 'bins_factor_range', 'threshold_range',
                   'width_range', 'prominence_range', 'distance_range']:
            if key in params:
                total *= len(params[key])
        return total
    
    fast_count = count_combinations(fast_params)
    thorough_count = count_combinations(thorough_params)
    
    print(f"\nPerformance comparison:")
    print(f"  Fast: {fast_count:,} combinations")
    print(f"  Thorough: {thorough_count:,} combinations")
    print(f"  Ratio: {thorough_count/fast_count:.1f}x more computation")
    
    print("\n✅ Hyperparameter examples complete!")
    print("\n💡 Tips:")
    print("  - Start with auto-optimization for first calibration")
    print("  - Use custom parameters for fine-tuning specific sources")
    print("  - Balance parameter coverage with computation time")
    print("  - Monitor total combinations to avoid excessive runtime")


if __name__ == "__main__":
    main()
