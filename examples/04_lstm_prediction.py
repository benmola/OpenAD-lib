"""
Example: LSTM Model for Biogas Prediction

This script demonstrates how to key a Long Short-Term Memory (LSTM) network
to predict biogas production from feedstock composition time-series data.
It loads data, trains the model, and plots the predictions.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from openad_lib.models.ml import LSTMModel
except ImportError:
    print("Error: PyTorch is likely not installed. Please install torch.")
    exit(1)

# Get the data directory path
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'openad_lib', 'data')

print("=" * 60)
print("LSTM Model for Biogas Prediction")
print("=" * 60)

# Check for sample data
lstm_data_path = os.path.join(DATA_DIR, 'sample_feedstock_timeseries.csv')

if os.path.exists(lstm_data_path):
    # Load the data
    data = pd.read_csv(lstm_data_path).dropna()
    print(f"\nLoaded {len(data)} samples from biogas plant data")
    
    # Define features and target
    features = ['Maize', 'Wholecrop', 'Chicken Litter', 'Lactose', 'Apple Pomace', 'Rice bran']
    target = 'Total_Biogas'
    
    X = data[features].values
    y = data[target].values
    
    print(f"  Features: {features}")
    print(f"  Target: {target}")
    
    # Split data (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create and train model
    print("\nTraining LSTM model (20 epochs)...")
    lstm = LSTMModel(input_dim=len(features), hidden_dim=24, output_dim=1)
    lstm.fit(X_train, y_train, epochs=100, verbose=True)
    
    # Evaluate
    metrics = lstm.evaluate(X_test, y_test)
    print(f"\nLSTM Test Metrics:")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  MAE: {metrics['mae']:.2f}")
    print(f"  R²: {metrics['r2']:.3f}")
    
    # Make Predictions for Plotting
    y_pred = lstm.predict(X_test)
    
    # Plotting
    print("\nGenerating prediction plot...")
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual Biogas', color='blue', alpha=0.7)
    plt.plot(y_pred, label='Predicted Biogas (LSTM)', color='red', linestyle='--')
    plt.title("LSTM Model Predictions vs Actual Data (Test Set)")
    plt.xlabel("Sample Index (Time)")
    plt.ylabel("Biogas Production")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
else:
    print(f"Error: Data file not found in {DATA_DIR}")
    print("Running in demo mode with synthetic data...")
    
    # Create synthetic data
    np.random.seed(42)
    t = np.linspace(0, 100, 200)
    X_demo = np.sin(t).reshape(-1, 1) + np.random.normal(0, 0.1, (200, 1))
    y_demo = (X_demo * 2 + 5).flatten()
    
    split = 160
    X_train, X_test = X_demo[:split], X_demo[split:]
    y_train, y_test = y_demo[:split], y_demo[split:]
    
    lstm = LSTMModel(input_dim=1, hidden_dim=10, output_dim=1)
    lstm.fit(X_train, y_train, epochs=10, verbose=False)
    
    y_pred = lstm.predict(X_test)
    
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title("Synthetic Data LSTM Demo")
    plt.legend()
    plt.show()

print("\nExample complete!")
