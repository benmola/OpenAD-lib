"""
Quick Test Script for AM2 Model

This script verifies the AM2 model works correctly with the sample data.
"""

import os
import matplotlib.pyplot as plt
from openad_lib.models.mechanistic import AM2Model, AM2Parameters

# Path to data
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'openad_lib', 'data')
am2_data_path = os.path.join(DATA_DIR, 'sample_AM2_data.csv')

print("=" * 60)
print("AM2 Model Test")
print("=" * 60)

# Create model with default calibrated parameters
model = AM2Model()

# Print the parameters
print("\nAM2 Model Parameters:")
print(f"  m1 (µ1max): {model.params.m1} d⁻¹")
print(f"  K1:         {model.params.K1} g COD/L")
print(f"  m2 (µ2max): {model.params.m2} d⁻¹")
print(f"  Ki:         {model.params.Ki} g COD/L")
print(f"  K2:         {model.params.K2} g COD/L")
print(f"  k1:         {model.params.k1}")
print(f"  k2:         {model.params.k2}")
print(f"  k3:         {model.params.k3}")
print(f"  k6:         {model.params.k6}")

# Load data
print(f"\nLoading data from: {am2_data_path}")
model.load_data(am2_data_path)
print(f"  Data points: {len(model.data)}")

# Run simulation
print("\nRunning simulation...")
results = model.run(verbose=True)

# Print metrics
print("\n")
model.print_metrics()

# Plot results
print("\nGenerating plots...")
model.plot_results(
    variables=['S1', 'S2', 'Q'],
    figsize=(12, 10),
    show_measured=True
)

print("\nTest complete!")
