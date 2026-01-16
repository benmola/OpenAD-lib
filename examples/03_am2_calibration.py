"""
Example: AM2 Model Calibration

This script demonstrates how to calibrate the AM2 model parameters using Optuna.
It optimizes the kinetic parameters to fit the provided experimental data
and plots the comparison between initial and calibrated models.

New in v0.2.0: Uses load_sample_data() for simplified data loading.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add src to path for development
current_dir = Path(__file__).parent.resolve()
src_path = current_dir.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import openad_lib as openad

print("=" * 60)
print("AM2 Model Calibration")
print("=" * 60)

# -------------------------------------------------------------------------
# 1. Load Data Using Library Utilities
# -------------------------------------------------------------------------
# Use built-in sample data loader - no hardcoded paths needed!
am2_data = openad.load_sample_data('am2_lab')
print(f"\nLoaded {len(am2_data)} samples from AM2 lab data")

# Initialize model and load data
model = openad.AM2Model()
model.load_data_from_dataframe(
    am2_data,
    S1in_col='SCODin',
    S1out_col='SCODout',
    S2out_col='VFAout',
    Q_col='Biogas'
)

# Run initial simulation
print("\nRunning initial simulation...")
initial_results = model.run(verbose=False)
initial_metrics = model.evaluate()

# Configure calibration
print("\nConfiguring calibration...")
calibrator = openad.AM2Calibrator(model)

# Define parameters to tune
params_to_tune = ['m1', 'K1', 'm2', 'Ki', 'K2']

# Define custom parameter bounds (optional - uses defaults if not specified)
param_bounds = {
    'm1': (0.01, 0.5),     # Growth rate of acidogens
    'K1': (5.0, 50.0),     # Half-saturation constant
    'm2': (0.1, 1.0),      # Growth rate of methanogens
    'Ki': (5.0, 50.0),     # Inhibition constant
    'K2': (10.0, 80.0)     # Half-saturation constant for VFA
}

# Define weights (focus on VFA and Biogas stability)
weights = {'S1': 0.5, 'S2': 1.0, 'Q': 1.0}

print(f"Parameters to tune: {params_to_tune}")
print(f"Optimization weights: {weights}")

# Run calibration
print("\nStarting optimization (50 trials)...")
best_params = calibrator.calibrate(
    params_to_tune=params_to_tune,
    param_bounds=param_bounds,
    n_trials=50,
    weights=weights,
    show_progress_bar=True
)

# Run final simulation
print("\nRunning simulated with calibrated parameters...")
final_results = model.run(verbose=False)

# Compare metrics
print("\nCalibration Improvement:")
final_metrics = model.evaluate()
variables = ['S1', 'S2', 'Q']
for var in variables:
    rmse_key = f"{var}_RMSE"
    if rmse_key in initial_metrics and rmse_key in final_metrics:
        initial_rmse = initial_metrics[rmse_key]
        final_rmse = final_metrics[rmse_key]
        imp = initial_rmse - final_rmse
        pct = (imp / initial_rmse) * 100 if initial_rmse != 0 else 0
        print(f"  {var} RMSE: {initial_rmse:.4f} -> {final_rmse:.4f} (Reduction: {pct:.1f}%)")


# Plot comparison
print("\nGenerating comparison plots...")

openad.plots.plot_calibration_comparison(
    initial_results, 
    final_results, 
    variables=['S1', 'S2', 'Q'],
    labels=['COD (S1)', 'VFA (S2)', 'Biogas (Q)'],
    save_plot=True,
    show=True
)

print("\nExample complete!")
