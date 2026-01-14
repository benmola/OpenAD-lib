"""
Example: AM2 Model Calibration using Optuna

This script demonstrates how to calibrate the AM2 model parameters
to fit experimental data using the Optuna optimization framework.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

try:
    import optuna
    from openad_lib.optimisation.am2_calibration import AM2Calibrator
    from openad_lib.models.mechanistic.am2_model import AM2Model
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure openad_lib and optuna are installed.")
    exit(1)

# =============================================================================
# 1. Setup Data and Model
# =============================================================================

# Path to sample data
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'openad_lib', 'data')
data_path = os.path.join(DATA_DIR, 'sample_AM2_data.csv')

print(f"Loading data from: {data_path}")

# Initialize model and load data
model = AM2Model()
model.load_data(data_path)

print(f"Data Loaded: {len(model.data)} points")
print("Initial Parameters:")
model.print_parameters()

# Run initial simulation to see baseline performance
print("\nRunning initial simulation...")
initial_results = model.run(verbose=False)
initial_metrics = model.evaluate()
print("\nInitial Metrics:")
for var, m in initial_metrics.items():
    print(f"  {var}: RMSE={m['RMSE']:.4f}, R2={m['R2']:.4f}")

# =============================================================================
# 2. Configure Calibration
# =============================================================================

print("\n" + "=" * 60)
print("Starting Calibration with Optuna")
print("=" * 60)

# Initialize calibrator
calibrator = AM2Calibrator(model=model)

# Define which parameters to tune (e.g., kinetic parameters)
# We'll tune the growth rates (m1, m2) and saturation constants (K1, Ki, K2)
params_to_tune = ['m1', 'K1', 'm2', 'Ki', 'K2']

# Define weights for the objective function (prioritize S2 and Q)
weights = {
    'S1': 3.0,  # COD
    'S2': 1.0,  # VFA (important for stability)
    'Q': 1.0    # Biogas (important for production)
}

# Run calibration
# Using a small number of trials for demonstration. In practice, use 50-100+.
n_trials = 100  
best_params = calibrator.calibrate(
    params_to_tune=params_to_tune,
    n_trials=n_trials,
    weights=weights,
    show_progress_bar=True
)

# =============================================================================
# 3. Analyze Results
# =============================================================================

print("\n" + "=" * 60)
print("Calibration Results")
print("=" * 60)

# The model has already been updated with best_params
print("Calibrated Parameters:")
model.print_parameters()

# Run simulation with calibrated parameters
final_results = model.run(verbose=False)
final_metrics = model.evaluate()

print("\nFinal Metrics (After Calibration):")
for var, m in final_metrics.items():
    print(f"  {var}: RMSE={m['RMSE']:.4f}, R2={m['R2']:.4f}")

# Compare improvement
print("\nImprovement:")
for var in initial_metrics:
    rmse_imp = initial_metrics[var]['RMSE'] - final_metrics[var]['RMSE']
    print(f"  {var} RMSE reduction: {rmse_imp:.4f} ({(rmse_imp/initial_metrics[var]['RMSE'])*100:.1f}%)")

# Plot comparison
print("\nGenerating comparison plots...")

# Create comparison plot
fig, axes = plt.subplots(3, 1, figsize=(10, 12))
variables = ['S1', 'S2', 'Q']
labels = ['COD (S1)', 'VFA (S2)', 'Biogas (Q)']

time = final_results['time']

for i, var in enumerate(variables):
    ax = axes[i]
    
    # Measured data
    if f'{var}_measured' in final_results.columns:
        valid_mask = ~final_results[f'{var}_measured'].isna()
        ax.plot(
            time[valid_mask], 
            final_results[f'{var}_measured'][valid_mask], 
            'bo', label='Measured', alpha=0.6
        )
    
    # Initial Simulation
    ax.plot(
        time, initial_results[var], 
        'r--', label='Initial Model', alpha=0.5
    )
    
    # Calibrated Simulation
    ax.plot(
        time, final_results[var], 
        'g-', linewidth=2, label='Calibrated Model'
    )
    
    ax.set_ylabel(labels[i])
    ax.set_title(f'{labels[i]} Calibration')
    ax.legend()
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (days)')
plt.tight_layout()
plt.show()

print("\nExample complete!")
