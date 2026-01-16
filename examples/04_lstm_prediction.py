"""
LSTM Prediction Example
=======================

This script demonstrates LSTM-based biogas prediction using the unified MLModel interface.

Workflow:
1. Load Data: Standard CSV format.
2. Initialize: Use LSTMModel (MLModel).
3. Preprocess: Use built-in model.prepare_time_series_data() for lags.
4. Train: Use unified train() method (handles scaling automatically).
5. Evaluate: Use unified evaluate() method.

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
    import openad_lib as oad
    from openad_lib.utils.metrics import print_metrics
except ImportError as e:
    print(f"Error importing openad_lib: {e}")
    sys.exit(1)

def main():
    print("="*60)
    print("LSTM Model for Biogas Prediction (Simplified)")
    print("="*60)

    # Paths
    base_data_path = src_path / 'openad_lib' / 'data'
    data_path = base_data_path / 'sample_LSTM_timeseries.csv'

    # Load data
    if data_path.exists():
        data = pd.read_csv(data_path).dropna()
        print(f"\nLoaded {len(data)} samples from biogas plant data")
        
        features = ['Maize', 'Wholecrop', 'Chicken Litter', 'Lactose', 'Apple Pomace', 'Rice bran']
        label = 'Total_Biogas'
        
        print(f"  Features: {features}")
        print(f"  Target: {label}")
    else:
        print(f"Error: Data file not found at {data_path}")
        return

    # Initialize Model (Needed first to use data prep helper)
    # We'll determine input_dim dynamically from the prepared data or features
    # But for now, let's initialize it placeholder or just use class method?
    # prepare_time_series_data is an instance method, so we instantiate first.
    
    # We don't know input_dim yet (it depends on lags).
    # Let's set it to 1 initially validation will fix it or we re-init?
    # Actually, proper flow: prepare data -> determine dims -> init model.
    # But to access helper we need instance.
    # Let's clean this up by instantiating with dummy dims, then updating?
    # OR we use the static method logic if exposed?
    # prepare_time_series_data is an instance method I added.
    
    # Let's just init with ANY dim, and update it before building network?
    # The network is built in __init__.
    # This is a slight design flaw in my "helper in model" approach if __init__ builds net immediately.
    # However, for this example, we can calculate expected dim easily: len(features) * n_lags.
    
    n_in = 1  # Previous timestep
    n_out = 1 # Not used for features, but for target horizon if needed
    
    input_dim = len(features) * n_in
    print(f"  Input Dimension: {input_dim} (Features * Lags)")

    # Initialize LSTM model (using simplified API)
    lstm = oad.LSTMModel(input_dim=input_dim, hidden_dim=24, output_dim=1)

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

    # Train (Model handles scaling internally!)
    print("\nTraining LSTM model...")
    lstm.train(
        X_train, 
        y_train, 
        epochs=50, 
        verbose=True
    )

    # Evaluate
    print("\nLSTM Evaluation Metrics (Test Set):")
    # evaluate() handles scaling (inverse transforms internally)
    metrics = lstm.evaluate(X_test, y_test)
    openad.utils.metrics.print_metrics(metrics, title="LSTM Test Performance")

    # Predict for plotting
    train_pred = lstm.predict(X_train)
    test_pred = lstm.predict(X_test)
    
    # Note: Model scaler is fitted during training.
    # We need to inverse transform the TRUE y values for plotting comparison,
    # because 'y_train' passed to plot should ideally be in original scale.
    # 'y_train' passed to fit IS unscaled (raw).
    # 'y_train' we have here IS unscaled (raw).
    # So we can just plot y_train directly!
    # Wait, predict() returns inverse_transformed (original scale) values by default?
    # Let's check model implementation:
    # predict() -> returns self.scaler_y.inverse_transform(predictions). YES.
    
    # So we compare raw y vs predict() output. Simple!

    # Plotting
    print("\nGenerating prediction plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(y_train, label='Actual', color='blue', alpha=0.7)
    ax1.plot(train_pred, label='Predicted', color='red', linestyle='--')
    ax1.set_title("Training Set")
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Biogas Production")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(y_test, label='Actual', color='blue', alpha=0.7)
    ax2.plot(test_pred, label='Predicted', color='red', linestyle='--')
    ax2.set_title("Testing Set")
    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("Biogas Production")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    images_dir = current_dir.parent / 'images'
    images_dir.mkdir(exist_ok=True)
    save_path = images_dir / 'lstm_prediction_result.png'
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    main()
