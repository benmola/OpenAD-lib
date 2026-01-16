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

try:
    import optuna
    import openad_lib as openad
    from openad_lib.optimisation import AM2Calibrator
    from openad_lib.models.mechanistic import AM2Model
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure openad_lib[optimization] is installed.")
    sys.exit(1)

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
model = AM2Model()
model.load_data(am2_data)

# Run initial simulation
print("\nRunning initial simulation...")
initial_results = model.run(verbose=False)
initial_metrics = model.evaluate()

# Configure calibration
print("\nConfiguring calibration...")
calibrator = AM2Calibrator(model)

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
images_dir = current_dir.parent / 'images'
images_dir.mkdir(exist_ok=True)

save_path = images_dir / 'am2_calibration.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {save_path}")

print("\nExample complete!")
