"""
Example: Multi-Task Gaussian Process (MTGP) for Process Prediction

This script demonstrates how to key a Multi-Task GP model to predict multiple
correlated outputs (e.g., SCOD, VFA, Biogas) simultaneously, providing
uncertainty estimates for each prediction.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from openad_lib.models.ml import MultitaskGP
except ImportError:
    print("Error: GPyTorch is likely not installed. Please install gpytorch.")
    exit(1)

# Get the data directory path
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'openad_lib', 'data')

print("=" * 60)
print("Multi-Task GP Prediction")
print("=" * 60)

# Check for sample data
mtgp_data_path = os.path.join(DATA_DIR, 'sample_ad_process_data.csv')

if os.path.exists(mtgp_data_path):
    # Load the data
    data = pd.read_csv(mtgp_data_path)
    print(f"\nLoaded {len(data)} samples from AD process data")
    
    # We will try to predict [SCODout, VFAout, Biogas] using [D, SCODin, OLR, pH]
    # This is a simplified example
    
    # Ensure columns exist (adjust based on your actual csv structure from previous steps)
    input_cols = ['D', 'SCODin', 'OLR', 'pH']
    target_tasks = ['SCODout', 'VFAout', 'Biogas']
    
    # Check if columns are present
    missing_cols = [c for c in input_cols + target_tasks if c not in data.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}. Using available numerical columns for demo.")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        # Split roughly
        n_inputs = len(numeric_cols) // 2
        input_cols = numeric_cols[:n_inputs]
        target_tasks = numeric_cols[n_inputs:n_inputs+3]
        print(f"Selected Inputs: {list(input_cols)}")
        print(f"Selected Targets: {list(target_tasks)}")
        
    X = data[input_cols].values
    Y = data[target_tasks].values
    
    # Split Train/Test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]
    
    # Initialize MTGP
    num_tasks = Y.shape[1]
    print(f"\nInitializing MTGP with {num_tasks} tasks...")
    mtgp = MultitaskGP(num_tasks=num_tasks, num_latents=min(3, num_tasks))
    
    # Train
    print("Training MTGP model (50 iterations)...")
    mtgp.fit(X_train, Y_train, epochs=50)
    
    # Predict
    print("\nPredicting on test set...")
    mean, lower, upper = mtgp.predict(X_test)
    
    # Plot results for each task
    print("Generating plots with uncertainty intervals...")
    fig, axes = plt.subplots(num_tasks, 1, figsize=(10, 4*num_tasks))
    if num_tasks == 1: axes = [axes]
    
    for i, task_name in enumerate(target_tasks):
        ax = axes[i]
        
        # Actual
        ax.plot(Y_test[:, i], 'b-', label='Actual')
        
        # Predicted Mean
        ax.plot(mean[:, i], 'r--', label='Predicted Mean')
        
        # Uncertainty
        ax.fill_between(
            range(len(Y_test)), 
            lower[:, i], 
            upper[:, i], 
            alpha=0.3, 
            color='red',
            label='95% Confidence Interval'
        )
        
        ax.set_title(f"Task: {task_name}")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.show()
    
else:
    print(f"Error: Data file not found in {DATA_DIR}")
    print("Please ensure sample_ad_process_data.csv exists.")

print("\nExample complete!")
