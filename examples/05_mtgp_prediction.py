"""
Multi-Task GP Prediction Example
================================

This script demonstrates Multi-Task Gaussian Process for multi-output prediction
using the unified MLModel interface.

Workflow:
1. Load Data: Standard CSV format.
2. Initialize: Use MultitaskGP (MLModel).
3. Train: Use unified train() method.
4. Evaluate: Use unified evaluate() method for multi-output metrics.

New in v0.2.0: Uses simplified API with top-level imports.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent.resolve()
src_path = current_dir.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    import openad_lib as openad
except ImportError as e:
    print(f"Error importing openad_lib: {e}")
    sys.exit(1)

def main():
    print("="*60)
    print("Multi-Task GP Prediction (Updated Structure)")
    print("="*60)

    # Paths
    base_data_path = src_path / 'openad_lib' / 'data'
    data_path = base_data_path / 'sample_ad_process_data.csv'

    # Load data
    if data_path.exists():
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
        print(f"Error: Data file not found at {data_path}")
        return

    # Split data using alternating indices (matching reference)
    train_indices = np.arange(1, len(X), 2)
    test_indices = np.arange(0, len(X), 2)
    
    X_train, X_test = X[train_indices], X[test_indices]
    Y_train, Y_test = Y[train_indices], Y[test_indices]
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Initialize MTGP using openad_lib
    # Initialize Multi-Task GP model (using simplified API)
    print(f"\nInitializing MTGP with {len(output_cols)} tasks...")
    mtgp = openad.MultitaskGP(
        num_tasks=len(output_cols),  # Note: parameter is 'num_tasks' not 'n_tasks'
        num_latents=min(3, len(output_cols)),
        n_inducing=60,
        learning_rate=0.1,
        log_transform=True  
    )
    
    # Train using unified API
    print("Training MTGP model (500 iterations)...")
    mtgp.train(X_train, Y_train, epochs=500, verbose=True)
    
    # Predict using unified API
    print("\nPredicting on test set...")
    # predict() returns mean, but we want std too for plot. 
    # Base.predict returns ndarray. Only MTGP specific predict handles return_std.
    # We can still use specific method if needed, but evaluate uses base predict.
    # Let's use the specific one for plotting confidence intervals.
    mean, lower, upper = mtgp.predict(X_test, return_std=True)
    
    # Evaluate using unified API
    print("\nEvaluation Metrics (Test Set):")
    # evaluate() calls predict() internally and computes metrics per task
    metrics = mtgp.evaluate(X_test, Y_test, task_names=output_cols)
    
    # Print metrics nicely
    # metrics format: {'SCODout': {'rmse': ..., 'r2': ...}, ...}
    for task_name, task_metrics in metrics.items():
        openad.utils.metrics.print_metrics(task_metrics, title=f"Task: {task_name}")
    
    # Plot using unified system
    print("Generating plots...")
    
    # Create indices
    n_train = len(X_train)
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, n_train + len(X_test))
    
    # Combine data
    X_full = np.vstack([X_train, X_test])
    Y_full = np.vstack([Y_train, Y_test])
    mean_full, lower_full, upper_full = mtgp.predict(X_full, return_std=True)
    
    # Save plot
    images_dir = current_dir.parent / 'images'
    images_dir.mkdir(exist_ok=True)
    save_path = images_dir / 'mtgp_prediction_result.png'
    
    openad.plots.plot_multi_output(
        y_true=Y_full,
        y_pred=mean_full,
        x=X_full[:, 0],
        y_lower=lower_full,
        y_upper=upper_full,
        train_indices=train_idx,
        test_indices=test_idx,
        output_names=output_cols,
        xlabel="Time",
        save_path=save_path,
        show=False
    )
    print(f"\nPlot saved to {save_path}")
    
    print("\nExample complete!")

if __name__ == "__main__":
    main()
