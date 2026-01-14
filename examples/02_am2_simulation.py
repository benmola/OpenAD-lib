"""
Example: AM2 Model Simulation (Simplified 4-State Model)

This script demonstrates how to use the simplified AM2 model.
It loads sample data, initializes the model with default parameters,
runs the simulation, and plots the results.
"""

import os
import matplotlib.pyplot as plt
from openad_lib.models.mechanistic import AM2Model, AM2Parameters

# Get the data directory path
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'openad_lib', 'data')

print("=" * 60)
print("AM2 Model Simulation")
print("=" * 60)

# Check if data file exists
am2_data_path = os.path.join(DATA_DIR, 'sample_AM2_Lab_data.csv')

if os.path.exists(am2_data_path):
    print("\nInitializing AM2 model with default calibrated parameters...")
    model = AM2Model()
    
    # Display default parameters
    print("\nDefault AM2 Parameters:")
    print(f"  µ1max (m1): {model.params.m1} d⁻¹")
    print(f"  K1:         {model.params.K1} g COD/L")
    print(f"  µ2max (m2): {model.params.m2} d⁻¹")
    print(f"  Ki:         {model.params.Ki} g COD/L")
    print(f"  K2:         {model.params.K2} g COD/L")
    
    # Load data
    print("\nLoading AM2 data...")
    model.load_data(am2_data_path)
    
    print("\nRunning AM2 simulation...")
    results = model.run(verbose=True)
    
    print("\nEvaluation Metrics:")
    model.print_metrics()
    
    print("\nGenerating plots...")
    model.plot_results(figsize=(10, 8), show_measured=True)
    
else:
    print(f"Error: Data file not found in {DATA_DIR}")
    print("Please ensure sample_AM2_Lab_data.csv exists.")
