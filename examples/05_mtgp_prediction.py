"""
Multi-Task GP Prediction Example
================================

This script demonstrates Multi-Task Gaussian Process for multi-output prediction
using the unified MLModel interface.

Workflow:
1. Load Data: Use load_sample_data() for built-in sample data.
2. Initialize: Use MultitaskGP (MLModel).
3. Train: Use unified train() method.
4. Evaluate: Use unified evaluate() method for multi-output metrics.

New in v0.2.0: Uses simplified API with top-level imports and load_sample_data().
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path for development
current_dir = Path(__file__).parent.resolve()
src_path = current_dir.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import openad_lib as openad

def main():
    print("="*60)
    print("Multi-Task GP Prediction (Updated Structure)")
    print("="*60)

    # -------------------------------------------------------------------------
    # 1. Load Data Using Library Utilities
    # -------------------------------------------------------------------------
    # Use built-in sample data loader - no hardcoded paths needed!
    data = openad.load_sample_data('mtgp')
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

    # -------------------------------------------------------------------------
    # 2. Split Data
    # -------------------------------------------------------------------------
    # Split data using alternating indices (matching reference)
    train_indices = np.arange(1, len(X), 2)
    test_indices = np.arange(0, len(X), 2)
    
    X_train, X_test = X[train_indices], X[test_indices]
    Y_train, Y_test = Y[train_indices], Y[test_indices]
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # -------------------------------------------------------------------------
    # 3. Initialize & Train MTGP
    # -------------------------------------------------------------------------
    print(f"\nInitializing MTGP with {len(output_cols)} tasks...")
    mtgp = openad.MultitaskGP(
        num_tasks=len(output_cols),
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
    mean, lower, upper = mtgp.predict(X_test, return_std=True)
    
    # Evaluate using unified API
    print("\nEvaluation Metrics (Test Set):")
    metrics = mtgp.evaluate(X_test, Y_test, task_names=output_cols)
    
    # Print metrics nicely
    for task_name, task_metrics in metrics.items():
        openad.utils.metrics.print_metrics(task_metrics, title=f"Task: {task_name}")

    # -------------------------------------------------------------------------
    # 4. Plot using unified system
    # -------------------------------------------------------------------------
    print("Generating plots...")
    
    # Create indices
    n_train = len(X_train)
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, n_train + len(X_test))
    
    # Combine data
    X_full = np.vstack([X_train, X_test])
    Y_full = np.vstack([Y_train, Y_test])
    mean_full, lower_full, upper_full = mtgp.predict(X_full, return_std=True)
    
    # Save plot - auto-save enabled!
    openad.plots.plot_multi_output(
        y_true=Y_full,
        y_pred=mean_full,
        x=X_full[:, 0],
        y_lower=lower_full,
        y_upper=upper_full,
        train_indices=train_idx,
        test_indices=test_idx,
        output_names=output_cols,
        title="Multi-Task GP Prediction",
        xlabel="Time",
        save_plot=True,
        show=True
    )
    
    print("\nExample complete!")

if __name__ == "__main__":
    main()
