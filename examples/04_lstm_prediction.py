"""
LSTM Prediction Example
=======================

This script demonstrates LSTM-based biogas prediction using the unified MLModel interface.

Workflow:
1. Load Data: Use load_sample_data() for built-in sample data.
2. Initialize: Use LSTMModel (MLModel).
3. Preprocess: Use built-in model.prepare_time_series_data() for lags.
4. Train: Use unified train() method (handles scaling automatically).
5. Evaluate: Use unified evaluate() method.

New in v0.2.0: Uses simplified API with top-level imports and load_sample_data().
"""

import sys
import numpy as np
from pathlib import Path
import pandas as pd
# Add src to path for development
current_dir = Path(__file__).parent.resolve()
src_path = current_dir.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import openad_lib as openad


def main():
    print("="*60)
    print("LSTM Model for Biogas Prediction (Simplified)")
    print("="*60)

    # -------------------------------------------------------------------------
    # 1. Load Data Using Library Utilities
    # -------------------------------------------------------------------------
    # Use built-in sample data loader - no hardcoded paths needed!
    data = openad.load_sample_data('lstm_timeseries').dropna()
    print(f"\nLoaded {len(data)} samples from biogas plant data")
    
    features = ['Maize', 'Wholecrop', 'Chicken Litter', 'Lactose', 'Apple Pomace', 'Rice bran']
    label = 'Total_Biogas'
    
    print(f"  Features: {features}")
    print(f"  Target: {label}")

    # -------------------------------------------------------------------------
    # 2. Initialize Model
    # -------------------------------------------------------------------------
    n_in = 1  # Previous timestep
    input_dim = len(features) * n_in
    print(f"  Input Dimension: {input_dim} (Features * Lags)")

    # Initialize LSTM model (using simplified API)
    lstm = openad.LSTMModel(input_dim=input_dim, hidden_dim=24, output_dim=1)

    # Prepare Data using Model Helper
    print("\nPreparing time series data (Creating lags)...")
    X, y, dataset = lstm.prepare_time_series_data(
        data, 
        features=features, 
        target=label, 
        n_in=n_in
    )

    # Split Data (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    # Flatten y for training (1D array)
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # -------------------------------------------------------------------------
    # 3. Train & Evaluate
    # -------------------------------------------------------------------------
    print("\nTraining LSTM model...")
    lstm.train(
        X_train, 
        y_train, 
        epochs=50, 
        verbose=True
    )

    # Evaluate
    print("\nLSTM Evaluation Metrics (Test Set):")
    metrics = lstm.evaluate(X_test, y_test)
    openad.utils.metrics.print_metrics(metrics, title="LSTM Test Performance")

    # Predict for plotting
    train_pred = lstm.predict(X_train)
    test_pred = lstm.predict(X_test)

    # -------------------------------------------------------------------------
    # 4. Plotting using unified system  
    # -------------------------------------------------------------------------
    print("\nGenerating prediction plot...")
    
    # Combine train and test for visualization
    y_full = np.concatenate([y_train, y_test])
    pred_full = np.concatenate([train_pred, test_pred])
    pd.DataFrame(pred_full, columns=['Predicted_Biogas']).to_csv('lstm_predictions.csv', index=False)
    n_train = len(y_train)
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, len(y_full))
    
    # Save plot - auto-save enabled!
    openad.plots.plot_predictions(
        y_true=y_full,
        y_pred=pred_full,
        train_indices=train_idx,
        test_indices=test_idx,
        title="LSTM Biogas Prediction",
        xlabel="Sample Index",
        ylabel="Biogas Production",
        save_plot=True,
        show=True
    )

if __name__ == "__main__":
    main()
