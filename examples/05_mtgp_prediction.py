"""
Multi-Task GP Prediction Example
================================

This script demonstrates MTGP-based multi-output prediction using openad_lib,
with data handling matching the reference implementation.

Features:
- Uses alternating train/test split (matching reference)
- Log-transform for outputs
- Uses openad_lib.models.ml.MultitaskGP

Usage:
    python examples/05_mtgp_prediction.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from openad_lib.models.ml import MultitaskGP
except ImportError as e:
    print(f"Error importing openad_lib: {e}")
    sys.exit(1)

def main():
    print("="*60)
    print("Multi-Task GP Prediction")
    print("="*60)

    # Paths
    base_dir = os.path.dirname(__file__)
    project_root = os.path.join(base_dir, '..')
    data_path = os.path.join(project_root, 'src', 'openad_lib', 'data', 'sample_ad_process_data.csv')

    # Load data
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
        print(f"\nLoaded {len(data)} samples from AD process data")
        
        # Define input and output columns explicitly
        input_cols = ['time', 'D', 'SCODin', 'OLR', 'pH']
        output_cols = ['SCODout', 'VFAout', 'Biogas']
        
        # Check if columns exist
        missing_inputs = [c for c in input_cols if c not in data.columns]
        missing_outputs = [c for c in output_cols if c not in data.columns]
        
        if missing_inputs or missing_outputs:
            print(f"Warning: Missing columns - Inputs: {missing_inputs}, Outputs: {missing_outputs}")
            print(f"Available columns: {data.columns.tolist()}")
            return
        
        X = data[input_cols].values
        Y = data[output_cols].values
        
        print(f"Input features: {input_cols}")
        print(f"Output tasks: {output_cols}")
    else:
        print("Error: Data file not found")
        return

    # Split data using alternating indices (matching reference)
    train_indices = np.arange(1, len(X), 2)
    test_indices = np.arange(0, len(X), 2)
    
    X_train, X_test = X[train_indices], X[test_indices]
    Y_train, Y_test = Y[train_indices], Y[test_indices]
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Initialize MTGP using openad_lib
    num_tasks = Y.shape[1]
    print(f"\nInitializing MTGP with {num_tasks} tasks...")
    mtgp = MultitaskGP(
        num_tasks=num_tasks,
        num_latents=min(3, num_tasks),
        n_inducing=60,
        learning_rate=0.1,
        log_transform=True  
    )
    
    # Train
    print("Training MTGP model (500 iterations)...")
    mtgp.fit(X_train, Y_train, epochs=500, verbose=True)
    
    # Predict
    print("\nPredicting on test set...")
    mean, lower, upper = mtgp.predict(X_test, return_std=True)
    
    # Evaluate
    metrics = mtgp.evaluate(X_test, Y_test, task_names=output_cols)
    for task, vals in metrics.items():
        print(f"{task}: RMSE={vals['rmse']:.4f}, MAE={vals['mae']:.4f}, R²={vals['r2']:.4f}")
    
    # Plot results
    print("Generating plots with uncertainty intervals...")
    fig, axes = plt.subplots(num_tasks, 1, figsize=(10, 4*num_tasks))
    if num_tasks == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        # Plot train and test data
        ax.plot(X_train[:, 0], Y_train[:, i], 'bo', label=f"True {output_cols[i]} (Train)", markersize=5, alpha=0.5)
        ax.plot(X_test[:, 0], Y_test[:, i], 'ro', label=f"True {output_cols[i]} (Test)", markersize=5, alpha=0.7)
        
        # Plot predictions with confidence
        ax.plot(X_test[:, 0], mean[:, i], color='black', label=f"Predicted {output_cols[i]}", linewidth=2)
        ax.fill_between(X_test[:, 0], lower[:, i], upper[:, i], color='black', alpha=0.2, label="95% Confidence")
        
        ax.set_title(output_cols[i])
        ax.set_xlabel("Time" if i == num_tasks-1 else "")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    images_dir = os.path.join(project_root, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    save_path = os.path.join(images_dir, 'mtgp_prediction_result.png')
    plt.savefig(save_path)
    print(f"\nPlot saved to {save_path}")
    
    print("\nExample complete!")

if __name__ == "__main__":
    main()
