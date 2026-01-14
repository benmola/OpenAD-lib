"""
Example: AM2 Model Calibration

This script demonstrates how to calibrate the AM2 model parameters using Optuna.
It optimizes the kinetic parameters to fit the provided experimental data
and plots the comparison between initial and calibrated models.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import optuna
    from openad_lib.optimisation import AM2Calibrator
    from openad_lib.models.mechanistic import AM2Model
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure openad_lib[optimization] is installed.")
    exit(1)

# Get the data directory path
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'openad_lib', 'data')
data_path = os.path.join(DATA_DIR, 'sample_AM2_Lab_data.csv')

print("=" * 60)
print("AM2 Model Calibration")
print("=" * 60)

if not os.path.exists(data_path):
    print(f"Error: Data file not found in {DATA_DIR}")
    exit(1)

# Initialize model and load data
print(f"Loading data from: {data_path}")
model = AM2Model()
model.load_data(data_path)

# Run initial simulation
print("\nRunning initial simulation...")
initial_results = model.run(verbose=False)
initial_metrics = model.evaluate()

# Configure calibration
print("\nConfiguring calibration...")
calibrator = AM2Calibrator(model)

# Define parameters to tune
params_to_tune = ['m1', 'K1', 'm2', 'Ki', 'K2']

# Define weights (focus on VFA and Biogas stability)
weights = {'S1': 0.5, 'S2': 1.0, 'Q': 1.0}

print(f"Parameters to tune: {params_to_tune}")
print(f"Optimization weights: {weights}")

# Run calibration
print("\nStarting optimization (50 trials)...")
best_params = calibrator.calibrate(
    params_to_tune=params_to_tune,
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
for var in initial_metrics:
    initial_rmse = initial_metrics[var]['RMSE']
    final_rmse = final_metrics[var]['RMSE']
    imp = initial_rmse - final_rmse
    pct = (imp / initial_rmse) * 100
    print(f"  {var} RMSE: {initial_rmse:.4f} -> {final_rmse:.4f} (Reduction: {pct:.1f}%)")

# Plot comparison
print("\nGenerating comparison plots...")
plt.style.use('bmh')
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
variables = ['S1', 'S2', 'Q']
labels = ['COD (S1)', 'VFA (S2)', 'Biogas (Q)']
time = final_results['time']

for i, var in enumerate(variables):
    ax = axes[i]
    
    # Measured Data
    if f'{var}_measured' in final_results.columns:
        valid = ~final_results[f'{var}_measured'].isna()
        ax.plot(time[valid], final_results[f'{var}_measured'][valid], 
                'o', color='#2E86C1', markersize=6, label='Measured', alpha=0.7)
        
    # Initial Model
    ax.plot(time, initial_results[var], '--', color='gray', linewidth=2, 
            label='Initial Model', alpha=0.7)
    
    # Calibrated Model
    ax.plot(time, final_results[var], '-', color='#27AE60', linewidth=2, 
            label='Calibrated Model')
    
    ax.set_ylabel(labels[i], fontsize=14, fontweight='bold')
    ax.set_title(f'{labels[i]} Calibration Comparison', fontsize=16, pad=20)
    ax.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='gray')
    ax.grid(True, linestyle='--', alpha=0.7)

axes[-1].set_xlabel('Time (days)', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save to images folder
base_dir = os.path.dirname(__file__)
project_root = os.path.join(base_dir, '..')
images_dir = os.path.join(project_root, 'images')
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

save_path = os.path.join(images_dir, 'am2_calibration.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {save_path}")

print("\nExample complete!")
